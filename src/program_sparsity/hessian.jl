# TermCombination datastructure

struct TermCombination
    terms::Set{Dict{Int, Int}} # idx => pow
end

@eval Base.zero(::Type{TermCombination}) = $(TermCombination(Set{Dict{Int,Int}}()))
@eval Base.one(::Type{TermCombination}) = $(TermCombination(Set([Dict{Int,Int}()])))

function Base.:(==)(comb1::TermCombination, comb2::TermCombination)
    comb1.terms == comb2.terms && return true

    n1 = reduce(max, (k for (k,_) in Iterators.flatten(comb1.terms)), init=0)
    n2 = reduce(max, (k for (k,_) in Iterators.flatten(comb2.terms)), init=0)
    n = max(n1, n2)

    _sparse(comb1, n) == _sparse(comb2, n)
end

function Base.:+(comb1::TermCombination, comb2::TermCombination)
    if isone(comb1) && !iszero(comb2)
        return comb2
    elseif isone(comb2) && !iszero(comb1)
        return comb1
    elseif comb1 === comb2
        return comb1
    end
    TermCombination(union(comb1.terms, comb2.terms))
end

Base.:+(comb1::TermCombination) = comb1

function _merge(dict1, dict2)
    d = copy(dict1)
    for (k, v) in dict2
        d[k] = min(2, get(dict1, k, 0) + v)
    end
    d
end

function Base.:*(comb1::TermCombination, comb2::TermCombination)
    if isone(comb1)
        return comb2
    elseif isone(comb2)
        return comb1
    elseif comb1 === comb2 # squaring optimization
        terms = comb1.terms
        # turns out it's enough to track
        # a^2*b^2
        # and a^2 + b^2 + ab
        # have the same hessian sparsity
        t = Dict(k=>2 for (k,_) in
                 Iterators.flatten(terms))
        TermCombination(Set([t]))
        #=
        # square each term
        t1 = [Dict(k=>2 for (k,_) in dict)
              for dict in comb1.terms]
        # multiply each term
        t2 = Dict{Int,Int}[]
        for i in 1:length(terms)
            for j in i+1:length(terms)
                push!(t2, _merge(terms[i], terms[j]))
            end
        end
        TermCombination(union(t1, t2))
        =#
    else
        Set([_merge(dict1, dict2)
             for dict1 in comb1.terms,
             dict2 in comb2.terms]) |> TermCombination
    end
end
Base.:*(comb1::TermCombination) = comb1
Base.iszero(c::TermCombination) = isempty(c.terms)
Base.isone(c::TermCombination) = all(isempty, c.terms)

function _sparse(t::TermCombination, n)
    I = Int[]
    J = Int[]
    for dict in t.terms
        kv = collect(pairs(dict))
        for i in 1:length(kv)
            k, v = kv[i]
            if v>=2
                push!(I, k)
                push!(J, k)
            end
            for j in i+1:length(kv)
                if v >= 1 && kv[j][2] >= 1
                    push!(I, k)
                    push!(J, kv[j][1])
                end
            end
        end
    end
    s1 = sparse(I,J,fill!(BitVector(undef, length(I)), true),n,n)
    s1 .| s1'
end

@proptagcontext HessianSparsityContext

istainted(ctx::HessianSparsityContext, val) = metatype(val, ctx) <: TermCombination

struct HessInput end

Cassette.metadatatype(::Type{<:HessianSparsityContext},
                      ::DataType) = Union{HessInput, TermCombination}
# optimization
Cassette.metadatatype(::Type{<:HessianSparsityContext},
                      ::Type{<:Number}) = TermCombination

haslinearity(ctx::HessianSparsityContext, f, nargs) = haslinearity(untag(f, ctx), nargs)
linearity(ctx::HessianSparsityContext, f, nargs) = linearity(untag(f, ctx), nargs)

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
function getterms(ctx, x)
    metatype(x, ctx) <: TermCombination ? metadata(x, ctx) : one(TermCombination)
end

function propagate_tags(ctx::HessianSparsityContext,
                        f, result, args...)

    if haslinearity(ctx, f, Val{nfields(args)}())
        l = linearity(ctx, f, Val{nfields(args)}())
        t = combine_terms(l, map(x->getterms(ctx, x), args)...)
        return tag(untag(result, ctx), ctx, t)
    else
        return result
    end
end

# getindex on the input
function Cassette.overdub(ctx::HessianSparsityContext,
                          f::typeof(getindex),
                          X::Tagged,
                          idx::Tagged...)
    if any(i->metatype(i, ctx) <: TermCombination && !isone(metadata(i, ctx)), idx)
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
    if metatype(X, ctx) <: HessInput
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
    if metatype(Y, ctx) <: HessInput
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
    if metatype(X, ctx) <: HessInput
        val = Cassette.fallback(ctx, f, X)
        tag(val, ctx, HessInput())
    else
        Cassette.recurse(ctx, f, X)
    end
end

function hessian_sparsity(f, X, args...; raw=false)

    terms = zero(TermCombination)
    ctx = HessianSparsityContext()
    ctx = Cassette.enabletagging(ctx, f)
    ctx = Cassette.disablehooks(ctx)
    val = nothing
    function process(result)
        try
            terms += metadata(result, ctx)
        catch err
            @warn("Could not extract hessian sparsity")
            println(err)
        end
        val=result
    end
    abstract_run(process,
                 ctx, f, tag(X, ctx, HessInput()),
                 map(arg -> arg isa Fixed ?
                     arg.value : tag(arg, ctx, one(TermCombination)), args)...)

    if raw
        return ctx, val
    end
    _sparse(terms, length(X))
end


# Forward BLAS calls to generic implementation
#
using LinearAlgebra
import LinearAlgebra.BLAS

# generic implementations

@reroute HessianSparsityContext BLAS.dot dot(Any, Any)
@reroute HessianSparsityContext BLAS.axpy! axpy!(Any,
                                                 AbstractArray,
                                                 AbstractArray)
