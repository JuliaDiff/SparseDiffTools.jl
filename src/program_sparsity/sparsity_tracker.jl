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

# Tags:
struct Input end
struct Output end

struct ProvinanceSet{T}
    set::T # Set, Array, Int, Tuple, anything!
end

# note: this is not strictly set union, just some efficient way of concating
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

Cassette.@context SparsityContext

const TagType = Union{Input, Output, ProvinanceSet}
Cassette.metadatatype(::Type{<:SparsityContext}, ::DataType) = TagType

metatype(x, ctx) = hasmetadata(x, ctx) && istagged(x, ctx) && typeof(metadata(x, ctx))
function ismetatype(x, ctx, T)
    hasmetadata(x, ctx) && istagged(x, ctx) && (metadata(x, ctx) isa T)
end

# Dummy type when you getindex
struct Tainted end

# getindex on the input
function Cassette.overdub(ctx::SparsityContext,
                          f::typeof(getindex),
                          X::Tagged,
                          idx::Int...)
    if ismetatype(X, ctx, Input)
        i = LinearIndices(untag(X, ctx))[idx...]
        val = Cassette.fallback(ctx, f, X, idx...)
        tag(val, ctx, ProvinanceSet(i))
    else
        Cassette.recurse(ctx, f, X, idx...)
    end
end

# setindex! on the output
function Cassette.overdub(ctx::SparsityContext,
                          f::typeof(setindex!),
                          Y::Tagged,
                          val::Tagged,
                          idx::Int...)
    S, path = ctx.metadata
    if ismetatype(Y, ctx, Output)
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

function get_provinance(ctx, arg::Tagged)
    if metadata(arg, ctx) isa ProvinanceSet
        metadata(arg, ctx)
    else
        ProvinanceSet(())
    end
end

get_provinance(ctx, arg) = ProvinanceSet(())

# Any function acting on a value tagged with ProvinanceSet
function _overdub_union_provinance(::Val{eval}, ctx::SparsityContext, f, args...) where {eval}
    idxs = findall(x->ismetatype(x, ctx, ProvinanceSet), args)
    if isempty(idxs)
        Cassette.fallback(ctx, f, args...)
    else
        provinance = union(map(arg->get_provinance(ctx, arg), args[idxs])...)
        if eval
            val = Cassette.fallback(ctx, f, args...)
            tag(val, ctx, provinance)
        else
            tag(Tainted(), ctx, provinance)
        end
    end
end

function Cassette.overdub(ctx::SparsityContext, f, args...)
    haspsets = any(x->ismetatype(x, ctx, ProvinanceSet), args)
    hasinput = any(x->ismetatype(x, ctx, Input), args)
    if haspsets && !hasinput # && !canrecurse(ctx, f, args...)
        _overdub_union_provinance(Val{true}(), ctx, f, args...)
    else
        Cassette.recurse(ctx, f, args...)
    end
end

#=
# Examples:
#
using UnicodePlots

sspy(s::Sparsity) = spy(sparse(s))

julia> sparsity!([0,0,0], [23,53,83]) do Y, X
           Y[:] .= X
           Y == X
       end
(true, Sparsity([1, 2, 3], [1, 2, 3]))

julia> sparsity!([0,0,0], [23,53,83]) do Y, X
           for i=1:3
               for j=i:3
                   Y[j] += X[i]
               end
           end; Y
       end
([23, 76, 159], Sparsity(3, 3, [1, 2, 3, 2, 3, 3], [1, 1, 1, 2, 2, 3]))

julia> sspy(ans[2])
     Sparsity Pattern
     ┌─────┐
   1 │⠀⠄⠀⠀⠀│ > 0
   3 │⠀⠅⠨⠠⠀│ < 0
     └─────┘
     1     3
     nz = 6

julia> sparsity!(f, zeros(Int, 3,3), [23,53,83])
([23, 53, 83], Sparsity(9, 3, [2, 5, 8], [1, 2, 3]))

julia> sspy(ans[2])
     Sparsity Pattern
     ┌─────┐
   1 │⠀⠄⠀⠀⠀│ > 0
     │⠀⠀⠠⠀⠀│ < 0
   9 │⠀⠀⠀⠐⠀│
     └─────┘
     1     3
     nz = 3
=#
