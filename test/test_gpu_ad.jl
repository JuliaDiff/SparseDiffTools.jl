using SparseDiffTools, CuArrays, Test, LinearAlgebra
using ArrayInterface: allowed_getindex, allowed_setindex!
function f(dx,x)
    dx[2:end-1] = x[1:end-2] - 2x[2:end-1] + x[3:end]
    allowed_setindex!(dx,-2allowed_getindex(x,1) + allowed_getindex(x,2),1)
    allowed_setindex!(dx,-2allowed_getindex(x,30) + allowed_getindex(x,29),30)
    nothing
end

_J1 = similar(rand(30,30))
_denseJ1 = cu(collect(_J1))
x = cu(rand(30))
CuArrays.allowscalar(false)
forwarddiff_color_jacobian!(_denseJ1, f, x)
forwarddiff_color_jacobian!(_denseJ1, f, x, sparsity = _J1)
forwarddiff_color_jacobian!(_denseJ1, f, x, colorvec = repeat(1:3,10), sparsity = _J1)
_Jt = similar(Tridiagonal(_J1))
@test_broken forwarddiff_color_jacobian!(_denseJ1, f, x, colorvec = repeat(1:3,10), sparsity = _Jt)
