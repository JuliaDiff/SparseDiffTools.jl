using Cassette
import Cassette: tag, untag, Tagged, metadata, hasmetadata, istagged, canrecurse
import Core: SSAValue
using SparseArrays

# Tags:
Cassette.@context HessianSparsityContext

const HTagType = Union{Input, TermCombination}
Cassette.metadatatype(::Type{<:HessianSparsityContext}, ::DataType) = HTagType

istainted(ctx::HessianSparsityContext, x) = ismetatype(x, ctx, TermCombination)

Cassette.overdub(ctx::HessianSparsityContext, f::typeof(istainted), x) = istainted(ctx, x)
Cassette.overdub(ctx::HessianSparsityContext, f::typeof(this_here_predicate!)) = this_here_predicate!(ctx.metadata)

# getindex on the input
function Cassette.overdub(ctx::HessianSparsityContext,
                          f::typeof(getindex),
                          X::Tagged,
                          idx::Int...)
    if ismetatype(X, ctx, Input)
        i = LinearIndices(untag(X, ctx))[idx...]
        val = Cassette.fallback(ctx, f, X, idx...)
        tag(val, ctx, TermCombination([Dict(i=>1)]))
    else
        Cassette.recurse(ctx, f, X, idx...)
    end
end

function Cassette.overdub(ctx::HessianSparsityContext,
                          f::typeof(Base.unsafe_copyto!),
                          X::Tagged,
                          xstart::Int,
                          Y::Tagged,
                          ystart::Int,
                          len::Int)
    if ismetatype(Y, ctx, Input)
        val = Cassette.fallback(ctx, f, X, xstart, Y, ystart, len)
        nometa = Cassette.NoMetaMeta()
        X.meta.meta[xstart:xstart+len-1] .= (i->Cassette.Meta(TermCombination([Dict(i=>1)]), nometa)).(ystart:ystart+len-1)
        val
    else
        Cassette.recurse(ctx, f, X, xstart, Y, ystart, len)
    end
end

combine_terms(::Nothing, terms...) = one(TermCombination)

# 1-arg functions
combine_terms(::Val{true}, term) = term
combine_terms(::Val{false}, term) = term * term

# 2-arg functions
function combine_terms(::Val{linearity}, term1, term2) where linearity

    linear11, linear22, linear12 = linearity
    term = zero(TermCombination)
    if linear11
        if !linear12
            term += term1
        end
    else
        term += term1 * term1
    end

    if linear22
        if !linear12
            term += term2
        end
    else
        term += term2 * term2
    end

    if linear12
        term += term1 + term2
    else
        term += term1 * term2
    end
    term
end


# Hessian overdub
#
function getterms(ctx, x)
    ismetatype(x, ctx, TermCombination) ? metadata(x, ctx) : one(TermCombination)
end

function hessian_overdub(ctx::HessianSparsityContext, f, linearity, args...)
    t = combine_terms(linearity, map(x->getterms(ctx, x), args)...)
    val = Cassette.fallback(ctx, f, args...)
    tag(val, ctx, t)
end
function Cassette.overdub(ctx::HessianSparsityContext,
                          f::typeof(getproperty),
                          x::Tagged, prop)
    if ismetatype(x, ctx, TermCombination) && !isone(metadata(x, ctx))
        Cassette.fallback(ctx, f, x, prop)
        error("property of a non-constant term accessed")
    else
        Cassette.fallback(ctx, f, x, prop)
    end
end

function Cassette.overdub(ctx::HessianSparsityContext,
                          f,
                          args...)
    if length(args) > 2
        return Cassette.recurse(ctx, f, args...)
    end
    tainted = any(x->ismetatype(x, ctx, TermCombination), args)
    if tainted && haslinearity(f, Val{nfields(args)}())
        l = linearity(f, Val{nfields(args)}())
        return hessian_overdub(ctx, f, l, args...)
    else
        val = Cassette.recurse(ctx, f, args...)
        if tainted && !ismetatype(val, ctx, TermCombination)
            error("Don't know the linearity of function $f")
        end
        return val
    end
end
