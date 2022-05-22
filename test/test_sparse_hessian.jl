
## Hessian tests
using SparsityDetection, SparseDiffTools
using LinearAlgebra, SparseArrays

function fscalar(x)
    return -dot(x, x)
end

x = randn(100)
hs = hessian_sparsity(fscalar, x)
col = matrix_colors(hs)
hescache = ForwardColorHesCache(fscalar, x, col)

# ForwardColorJacCache(f,x,_chunksize = nothing;
#                               dx = nothing,
#                               colorvec=1:length(x),
#                               sparsity = nothing)

# forwarddiff_color_hessian!(J::AbstractMatrix{<:Number},
#                             f,
#                             x::AbstractArray{<:Number};
#                             dx = nothing,
#                             colorvec = eachindex(x),
#                             sparsity = nothing)

# forwarddiff_color_jacobian!(J::AbstractMatrix{<:Number},
#                             f,
#                             x::AbstractArray{<:Number},
#                             jac_cache::ForwardColorJacCache)

# jacout = forwarddiff_color_jacobian(g, x,
#                                     dx = similar(x),
#                                     colorvec = 1:length(x),
#                                     sparsity = nothing,
#                                     jac_prototype = nothing) # matrix w/ sparsity pattern