using SparseArrays
using LinearAlgebra
using SparseDiffTools

struct ParaboloidStruct{T, Tm <: AbstractArray{T,2},
                           Tv <: AbstractArray{T}} <: Any where T<:Number
    mat::Tm
    vec::Tv
    xt::Tv
    alpha::T
end

function quad(x::Vector, param)
    mat = param.mat
    xt = x-param.vec
    return 0.5*dot(xt, mat*xt)
end

function _paraboloidproblem(N::Int;
                            mat::AbstractArray{T,2} = sparse(Diagonal(float(1:N))),
                            alpha::T=10.0,
                            x0::AbstractVector{T} = ones(N)) where T <: Number
    hsparsity(quad,x0,ParaboloidStruct(mat, x0, similar(x0), alpha))
end

@test isdiag(_paraboloidproblem(10))
