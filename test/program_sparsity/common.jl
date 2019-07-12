### Rules
using Test
using Cassette
import Cassette: tag, untag, Tagged, metadata, hasmetadata, istagged
using SparseArrays
using SparseDiffTools
import SparseDiffTools: abstract_run, HessianSparsityContext, JacobianSparsityContext

function jactester(f, Y, X, args...)
    ctx, val = jacobian_sparsity(f, Y, X, args...; raw=true)
end

jactestmeta(args...) = jactester(args...)[1].metadata
jactestval(args...) = jactester(args...) |> ((ctx,val),) -> untag(val, ctx)
jactesttag(args...) = jactester(args...) |> ((ctx,val),) -> metadata(val, ctx)

function hesstester(f, X, args...)
    ctx, val = hessian_sparsity(f, X, args...; raw=true)
end

hesstestmeta(args...) = hesstester(args...)[1].metadata
hesstestval(args...)  = hesstester(args...) |> ((ctx,val),) -> untag(val, ctx)
hesstesttag(args...)  = hesstester(args...) |> ((ctx,val),) -> metadata(val, ctx)


Base.show(io::IO, ::Type{<:Cassette.Context}) = print(io, "ctx")
Base.show(io::IO, ::Type{<:Tagged{<:Any, A}}) where {A} = print(io, "tagged{", A, "}")

