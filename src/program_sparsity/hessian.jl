using Cassette
import Cassette: tag, untag, Tagged, metadata, hasmetadata, istagged, canrecurse
import Core: SSAValue
using SparseArrays

# Tags:
Cassette.@context HessianSparsityContext

include("terms.jl")
const HTagType = Union{Input, TermCombination}
Cassette.metadatatype(::Type{<:HessianSparsityContext}, ::DataType) = HTagType

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

# linearity of a single input function is either
# Val{true}() or Val{false}()
#
# linearity of a 2-arg function is:
# Val{(linear11, linear22, linear12)}()
#
# linearIJ refers to the zeroness of d^2/dxIxJ

include("linearity.jl")
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
function getterms(ctx, x::Tagged)
    ismetatype(x, ctx, TermCombination) ? metadata(x, ctx) : one(TermCombination)
end

function hessian_overdub(ctx::HessianSparsityContext, f, linearity, args...)
    if any(x->ismetatype(x, ctx, TermCombination), args)
        t = combine_terms(linearity, map(x->getterms(ctx, x), args)...)
        val = Cassette.fallback(ctx, f, args...)
        tag(val, ctx, t)
    else
        Cassette.recurse(ctx, f, args...)
    end
end

function Cassette.overdub(ctx::HessianSparsityContext,
                          f,
                          args...)
    if haslinearity(f, Val{nfields(args)}())
        l = linearity(f, Val{nfields(args)}())
        return hessian_overdub(ctx, f, l, args...)
    else
        return Cassette.recurse(ctx, f, args...)
    end
end
