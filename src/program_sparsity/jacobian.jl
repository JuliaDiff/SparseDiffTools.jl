## Data structure for tracking sparsity

"""
The sparsity pattern.

- `I`: Input index
- `J`: Ouput index

`(i, j)` means the `j`th element of the output depends on
the `i`th element of the input. Therefore `length(I) == length(J)`
"""
struct Sparsity
    m::Int
    n::Int
    I::Vector{Int} # Input
    J::Vector{Int} # Output
end

SparseArrays.sparse(s::Sparsity) = sparse(s.I, s.J, true, s.m, s.n)

Sparsity(m, n) = Sparsity(m, n, Int[], Int[])

function Base.push!(S::Sparsity, i::Int, j::Int)
    push!(S.I, i)
    push!(S.J, j)
end

struct ProvinanceSet{T}
    set::T # Set, Array, Int, Tuple, anything!
end

# note: this is not strictly set union, just some efficient way of concating
Base.union(p::ProvinanceSet, ::Cassette.NoMetaData) = p
Base.union(::Cassette.NoMetaData, p::ProvinanceSet) = p

Base.union(p::ProvinanceSet{<:Tuple},
           q::ProvinanceSet{<:Integer}) = ProvinanceSet((p.set..., q.set,))
Base.union(p::ProvinanceSet{<:Integer},
           q::ProvinanceSet{<:Tuple}) = ProvinanceSet((p.set, q.set...,))
Base.union(p::ProvinanceSet{<:Integer},
           q::ProvinanceSet{<:Integer}) = ProvinanceSet((p.set, q.set,))
Base.union(p::ProvinanceSet{<:Tuple},
           q::ProvinanceSet{<:Tuple}) = ProvinanceSet((p.set..., q.set...,))
Base.union(p::ProvinanceSet,
           q::ProvinanceSet) = ProvinanceSet(union(p.set, q.set))
Base.union(p::ProvinanceSet,
           q::ProvinanceSet,
           rs::ProvinanceSet...) = union(union(p, q), rs...)
Base.union(p::ProvinanceSet) = p

function Base.push!(S::Sparsity, i::Int, js::ProvinanceSet)
    for j in js.set
        push!(S, i, j)
    end
end

# Cassette

@proptagcontext JacobianSparsityContext

struct JacInput end
struct JacOutput end

const TagType = Union{JacInput,
                      JacOutput,
                      ProvinanceSet}

istainted(ctx::JacobianSparsityContext, val) = metatype(val, ctx) <: ProvinanceSet

Cassette.metadatatype(::Type{<:JacobianSparsityContext},
                      ::DataType) = TagType
# optimization
Cassette.metadatatype(::Type{<:JacobianSparsityContext},
                      ::Type{<:Number}) = ProvinanceSet

function propagate_tags(ctx::JacobianSparsityContext,
                        f, result, args...)

    # e.g. X .+ X[1]
    # XXX: This wouldn't be required if we didn't have Input
    #
    any(x->metatype(x, ctx) <: JacInput,
        args) && return result

    idxs = findall(args) do x
        metatype(x, ctx) <: ProvinanceSet
    end

    if isempty(idxs)
        return result
    else
        tag(untag(result, ctx),
            ctx,
            union(map(x->metadata(x, ctx), args[idxs])...))
    end
end
const TaggedOf{T} = Cassette.Tagged{A, T} where A

function Cassette.overdub(ctx::JacobianSparsityContext,
                          f::typeof(getindex),
                          X::Tagged,
                          idx::Union{TaggedOf{Int},Int}...)
    if metatype(X, ctx) <: JacInput
        i = LinearIndices(untag(X, ctx))[idx...]
        val = Cassette.fallback(ctx, f, X, idx...)
        tag(val, ctx, ProvinanceSet(i))
    else
        Cassette.recurse(ctx, f, X, idx...)
    end
end

# setindex! on the output
function Cassette.overdub(ctx::JacobianSparsityContext,
                          f::typeof(setindex!),
                          Y::Tagged,
                          val::Tagged,
                          idx::Int...)
    S = ctx.metadata
    if metatype(Y, ctx) <: JacOutput
        set = metadata(val, ctx)
        if set isa ProvinanceSet
            i = LinearIndices(untag(Y, ctx))[idx...]
            push!(S, i, set)
        end
        Cassette.fallback(ctx, f, Y, val, idx...)
    else
        Cassette.recurse(ctx, f, Y, val, idx...)
    end
end

Base.@deprecate sparsity!(args...) jacobian_sparsity(args...)

function jacobian_sparsity(f!, Y, X, args...;
                           sparsity=Sparsity(length(Y), length(X)),
                           raw = false)

    ctx = JacobianSparsityContext(metadata=sparsity)
    ctx = Cassette.enabletagging(ctx, f!)
    ctx = Cassette.disablehooks(ctx)

    res = nothing
    abstract_run((result)->(res=result),
                 ctx,
                 f!,
                 tag(Y, ctx, JacOutput()),
                 tag(X, ctx, JacInput()),
                 map(arg -> arg isa Fixed ?
                     arg.value : tag(arg, ctx, ProvinanceSet(())), args)...)

    if raw
        return (ctx, res)
    else
        return sparse(sparsity)
    end
end
