# Tags:
Cassette.@context HessianSparsityContext

const TaggedOf{T} = Tagged{A, T} where A

const HTagType = Union{Input, TermCombination}
Cassette.metadatatype(::Type{<:HessianSparsityContext}, ::DataType) = HTagType

istainted(ctx::HessianSparsityContext, x) = ismetatype(x, ctx, TermCombination)

Cassette.overdub(ctx::HessianSparsityContext, f::typeof(istainted), x) = istainted(ctx, x)
Cassette.overdub(ctx::HessianSparsityContext, f::typeof(this_here_predicate!)) = this_here_predicate!(ctx.metadata)

# getindex on the input
function Cassette.overdub(ctx::HessianSparsityContext,
                          f::typeof(getindex),
                          X::Tagged,
                          idx::Tagged...)
    if any(i->ismetatype(i, ctx, TermCombination) && !isone(metadata(i, ctx)), idx)
        error("getindex call depends on input. Cannot determine Hessian sparsity")
    end
    Cassette.overdub(ctx, f, X, map(i->untag(i, ctx), idx)...)
end

# plugs an ambiguity
function Cassette.overdub(ctx::HessianSparsityContext,
                          f::typeof(getindex),
                          X::Tagged)
    Cassette.recurse(ctx, f, X)
end

function Cassette.overdub(ctx::HessianSparsityContext,
                          f::typeof(getindex),
                          X::Tagged,
                          idx::Integer...)
    if ismetatype(X, ctx, Input)
        val = Cassette.fallback(ctx, f, X, idx...)
        i = LinearIndices(untag(X, ctx))[idx...]
        tag(val, ctx, TermCombination(Set([Dict(i=>1)])))
    else
        Cassette.recurse(ctx, f, X, idx...)
    end
end

function Cassette.overdub(ctx::HessianSparsityContext,
                          f::typeof(Base.unsafe_copyto!),
                          X::Tagged,
                          xstart,
                          Y::Tagged,
                          ystart,
                          len)
    if ismetatype(Y, ctx, Input)
        val = Cassette.fallback(ctx, f, X, xstart, Y, ystart, len)
        nometa = Cassette.NoMetaMeta()
        X.meta.meta[xstart:xstart+len-1] .= (i->Cassette.Meta(TermCombination(Set([Dict(i=>1)])), nometa)).(ystart:ystart+len-1)
        val
    else
        Cassette.recurse(ctx, f, X, xstart, Y, ystart, len)
    end
end
function Cassette.overdub(ctx::HessianSparsityContext,
                          f::typeof(copy),
                          X::Tagged)
    if ismetatype(X, ctx, Input)
        val = Cassette.fallback(ctx, f, X)
        tag(val, ctx, Input())
    else
        Cassette.recurse(ctx, f, X)
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
        error("property of a non-constant term accessed")
    else
        Cassette.recurse(ctx, f, x, prop)
    end
end

haslinearity(ctx::HessianSparsityContext, f, nargs) = haslinearity(untag(f, ctx), nargs)
linearity(ctx::HessianSparsityContext, f, nargs) = linearity(untag(f, ctx), nargs)

function Cassette.overdub(ctx::HessianSparsityContext,
                          f,
                          args...)
    tainted = any(x->ismetatype(x, ctx, TermCombination), args)
    val = if tainted && haslinearity(ctx, f, Val{nfields(args)}())
        l = linearity(ctx, f, Val{nfields(args)}())
        hessian_overdub(ctx, f, l, args...)
    else
        val = Cassette.recurse(ctx, f, args...)
       #if tainted && !ismetatype(val, ctx, TermCombination)
       #    @warn("Don't know the linearity of function $f")
       #end
        val
    end
    val
end
