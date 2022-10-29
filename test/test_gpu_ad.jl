using SparseDiffTools, CUDA, Test, LinearAlgebra
using ArrayInterfaceCore: allowed_getindex, allowed_setindex!
using SparseArrays
using ArrayInterfaceGPUArrays

function f(dx, x)
    dx[2:(end - 1)] = x[1:(end - 2)] - 2x[2:(end - 1)] + x[3:end]
    allowed_setindex!(dx, -2allowed_getindex(x, 1) + allowed_getindex(x, 2), 1)
    allowed_setindex!(dx, -2allowed_getindex(x, 30) + allowed_getindex(x, 29), 30)
    nothing
end
x = rand(4)
_J1 = similar(rand(30, 30))
_denseJ1 = cu(collect(_J1))
x = cu(rand(30))
CUDA.allowscalar(false)
_J2 = sparse(forwarddiff_color_jacobian!(_denseJ1, f, x))
out = copy(_J2)
forwarddiff_color_jacobian!(out, f, x, colorvec = repeat(1:3, 10), sparsity = _J2)

@test_broken forwarddiff_color_jacobian!(_denseJ1, f, x, sparsity = cu(_J1)) isa Nothing
@test_broken forwarddiff_color_jacobian!(_denseJ1, f, x, colorvec = repeat(1:3, 10),
                                         sparsity = cu(_J1)) isa Nothing
_Jt = similar(Tridiagonal(_J1))
@test_broken forwarddiff_color_jacobian!(_denseJ1, f, x, colorvec = repeat(1:3, 10),
                                         sparsity = _Jt) isa Nothing
_Jt2 = similar(Tridiagonal(cu(_J1)))
@test_broken forwarddiff_color_jacobian!(_denseJ1, f, x, colorvec = repeat(1:3, 10),
                                         sparsity = _Jt2) isa Nothing
