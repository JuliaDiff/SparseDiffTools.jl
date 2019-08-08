using SparseDiffTools
using ForwardDiff: Dual, jacobian
using SparseArrays, Test
using LinearAlgebra
using BlockBandedMatrices

fcalls = 0
function f(dx,x)
    global fcalls += 1
    for i in 2:length(x)-1
        dx[i] = x[i-1] - 2x[i] + x[i+1]
    end
    dx[1] = -2x[1] + x[2]
    dx[end] = x[end-1] - 2x[end]
    nothing
end

function second_derivative_stencil(N)
    A = zeros(N,N)
    for i in 1:N, j in 1:N
        (j-i==-1 || j-i==1) && (A[i,j]=1)
        j-i==0 && (A[i,j]=-2)
    end
    A
end

x = rand(30)
dx = rand(30)

J = jacobian(f, dx, x)
@test J ≈ second_derivative_stencil(30)
_J = sparse(J)
@test fcalls == 3

fcalls = 0
_J1 = similar(_J)
forwarddiff_color_jacobian!(_J1, f, x, colorvec = repeat(1:3,10))
@test _J1 ≈ J
@test fcalls == 1

fcalls = 0
jac_cache = ForwardColorJacCache(f,x,colorvec = repeat(1:3,10), sparsity = _J1)
forwarddiff_color_jacobian!(_J1, f, x, jac_cache)
@test _J1 ≈ J
@test fcalls == 1

fcalls = 0
_J1 = similar(_J)
_denseJ1 = collect(_J1)
forwarddiff_color_jacobian!(_denseJ1, f, x, colorvec = repeat(1:3,10), sparsity = _J1)
@test _denseJ1 ≈ J
@test fcalls == 1

fcalls = 0
_J1 = similar(_J)
_denseJ1 = collect(_J1)
jac_cache = ForwardColorJacCache(f,x,colorvec = repeat(1:3,10), sparsity = _J1)
forwarddiff_color_jacobian!(_denseJ1, f, x, jac_cache)
@test _denseJ1 ≈ J
@test fcalls == 1

_Jt = similar(Tridiagonal(J))
forwarddiff_color_jacobian!(_Jt, f, x, colorvec = repeat(1:3,10), sparsity = _Jt)
@test _Jt ≈ J

#https://github.com/JuliaDiffEq/DiffEqDiffTools.jl/issues/67#issuecomment-516871956
function f(out, x)
	x = reshape(x, 100, 100)
	out = reshape(out, 100, 100)
	for i in 1:100
		for j in 1:100
			out[i, j] = x[i, j] + x[max(i -1, 1), j] + x[min(i+1, size(x, 1)), j] +  x[i, max(j-1, 1)]  + x[i, min(j+1, size(x, 2))]
		end
	end
	return vec(out)
end
x = rand(10000)
J = BandedBlockBandedMatrix(Ones(10000, 10000), (fill(100, 100), fill(100, 100)), (1, 1), (1, 1))
Jsparse = sparse(J)
colors = matrix_colors(J)
forwarddiff_color_jacobian!(J, f, x, colorvec=colors)
forwarddiff_color_jacobian!(Jsparse, f, x, colorvec=colors)
@test J ≈ Jsparse