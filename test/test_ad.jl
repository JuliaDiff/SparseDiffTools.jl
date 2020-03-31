using SparseDiffTools
using ForwardDiff: Dual, jacobian
using SparseArrays, Test
using LinearAlgebra
using BlockBandedMatrices
using StaticArrays

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

function oopf(x)
    global fcalls += 1
    dx = zero(x)
    for i in 2:length(x)-1
        dx[i] = x[i-1] - 2x[i] + x[i+1]
    end
    dx[1] = -2x[1] + x[2]
    dx[end] = x[end-1] - 2x[end]
    dx
end

function nsqf(x)#length(dx)<length(x)
    global fcalls +=1
    dx = zero(x)[1:div(length(x),2)]
    for i in 2:length(dx)
        dx[i] = x[i-1] - 2x[i] + x[i+1]
    end
    dx[1] = -2x[1] + x[2]
    dx
end

function nsqf2(x)#length(dx)>length(x)
    global fcalls +=1
    dx = zeros(eltype(x),length(x)*2)
    for i in 2:length(x)-1
        dx[i] = x[i-1] - 2x[i] + x[i+1]
    end
    dx[1] = -2x[1] + x[2]
    dx
end

function nsqf!(dx,x)
    global fcalls +=1
    for i in 2:length(dx)
        dx[i] = x[i-1] - 2x[i] + x[i+1]
    end
    dx[1] = -2x[1] + x[2]
    nothing
end

function nsqf2!(dx,x)
    global fcalls +=1
    for i in 2:length(x)-1
        dx[i] = x[i-1] - 2x[i] + x[i+1]
    end
    dx[1] = -2x[1] + x[2]
    nothing
end

function staticf(x,N=length(x))
    global fcalls += 1
    SVector{N}([i == 1 ? -2x[1]+x[2] : (i == N ? x[N-1]-2x[N] : x[i-1]-2x[i]+x[i+1]) for i in 1:N])
end

function staticnsqf(x,N=div(length(x),2))
    global fcalls += 1
    SVector{N}(vcat([-2x[1]+x[2]],[x[i-1]-2x[i]+x[i+1] for i in 2:N]))
end

function second_derivative_stencil(N)
    A = zeros(N,N)
    for i in 1:N, j in 1:N
        (j-i==-1 || j-i==1) && (A[i,j]=1)
        j-i==0 && (A[i,j]=-2)
    end
    A
end

@info "ended definitions"

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

@info "second passed"

fcalls = 0
_J1 = forwarddiff_color_jacobian(oopf, x, colorvec = repeat(1:3,10), sparsity = _J, jac_prototype = _J)
@test _J1 ≈ J
@test typeof(_J1) == typeof(_J)
@test fcalls == 1

@info "third passed"

fcalls = 0
_J1 = forwarddiff_color_jacobian(oopf, x, colorvec = repeat(1:3,10), sparsity = _J)
@test _J1 ≈ J
@test fcalls == 1

@info "4th passed"

fcalls = 0
_J1 = forwarddiff_color_jacobian(staticf, SVector{30}(x), colorvec = repeat(1:3,10), sparsity = _J, jac_prototype = SMatrix{30,30}(_J))
@test _J1 ≈ J
@test fcalls == 1

@info "5"

_J1 = forwarddiff_color_jacobian(staticf, SVector{30}(x), jac_prototype = SMatrix{30,30}(_J))
@test _J1 ≈ J
_J1 = forwarddiff_color_jacobian(oopf, x, jac_prototype = similar(_J))
@test _J1 ≈ J
_J1 = forwarddiff_color_jacobian(oopf, x)
@test _J1 ≈ J

#Non-square Jacobian
#length(dx)<length(x)
nsqJ = jacobian(nsqf,x)
spnsqJ = sparse(nsqJ)
_nsqJ = forwarddiff_color_jacobian(nsqf, x, dx = nothing)
@test _nsqJ ≈ nsqJ
_nsqJ = forwarddiff_color_jacobian(nsqf, x, colorvec = repeat(1:3,10), sparsity = spnsqJ)
@test _nsqJ ≈ nsqJ
_nsqJ = forwarddiff_color_jacobian(nsqf, x, jac_prototype = similar(nsqJ))
@test _nsqJ ≈ nsqJ
_nsqJ = forwarddiff_color_jacobian(nsqf, x, colorvec = repeat(1:3,10), sparsity = spnsqJ, jac_prototype = similar(nsqJ))
@test _nsqJ ≈ nsqJ
_nsqJ = forwarddiff_color_jacobian(nsqf, x, jac_prototype = SMatrix{15,30}(nsqJ))
@test _nsqJ ≈ nsqJ
@test typeof(_nsqJ) == typeof(SMatrix{15,30}(nsqJ))
_nsqJ = forwarddiff_color_jacobian(staticnsqf, SVector{30}(x), jac_prototype = SMatrix{15,30}(nsqJ))
@test _nsqJ ≈ nsqJ
_nsqJ = forwarddiff_color_jacobian(staticnsqf, SVector{30}(x), jac_prototype = SMatrix{15,30}(nsqJ), colorvec = repeat(1:3,10), sparsity = spnsqJ)
@test _nsqJ ≈ nsqJ
_nsqJ = similar(nsqJ)
forwarddiff_color_jacobian!(_nsqJ, nsqf!, x)
@test _nsqJ ≈ nsqJ
_nsqJ = similar(nsqJ)
forwarddiff_color_jacobian!(_nsqJ, nsqf!, x, colorvec = repeat(1:3,10), sparsity = spnsqJ )
@test _nsqJ ≈ nsqJ

#length(dx)>length(x)
nsqJ = jacobian(nsqf2,x)
spnsqJ = sparse(nsqJ)
_nsqJ = forwarddiff_color_jacobian(nsqf2, x, dx = nothing)
@test _nsqJ ≈ nsqJ
_nsqJ = forwarddiff_color_jacobian(nsqf2, x, colorvec = repeat(1:3,10), sparsity = spnsqJ)
@test _nsqJ ≈ nsqJ
_nsqJ = forwarddiff_color_jacobian(nsqf2, x, jac_prototype = similar(nsqJ))
@test _nsqJ ≈ nsqJ
_nsqJ = forwarddiff_color_jacobian(nsqf2, x, colorvec = repeat(1:3,10), sparsity = spnsqJ, jac_prototype = similar(nsqJ))
@test _nsqJ ≈ nsqJ
_nsqJ = forwarddiff_color_jacobian(nsqf2, x, jac_prototype = SMatrix{60,30}(nsqJ))
@test _nsqJ ≈ nsqJ
_nsqJ = similar(nsqJ)
forwarddiff_color_jacobian!(_nsqJ, nsqf2!, x)
@test _nsqJ ≈ nsqJ
_nsqJ = similar(nsqJ)
forwarddiff_color_jacobian!(_nsqJ, nsqf2!, x, colorvec = repeat(1:3,10), sparsity = spnsqJ )
@test _nsqJ ≈ nsqJ

fcalls = 0
_J1 = similar(_J)
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

#https://github.com/JuliaDiff/FiniteDiff.jl/issues/67#issuecomment-516871956
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
J = BandedBlockBandedMatrix(Ones(10000, 10000), fill(100, 100), fill(100, 100), (1, 1), (1, 1))
Jsparse = sparse(J)
colors = matrix_colors(J)
forwarddiff_color_jacobian!(J, f, x, colorvec=colors)
forwarddiff_color_jacobian!(Jsparse, f, x, colorvec=colors)
@test J ≈ Jsparse

# Non vector input
x = rand(2,2)
oopf(x) = x
iipf(fx,x) = (fx.=x)
J = forwarddiff_color_jacobian(oopf,x)
@test J ≈ Matrix(I,4,4)
J = zero(J)
forwarddiff_color_jacobian!(J,iipf,x,dx=similar(x))
@test J ≈ Matrix(I,4,4)

#1x1 SVector test
x = SVector{1}([1.])
f(x) = x
J = forwarddiff_color_jacobian(f,x)
@test J isa SArray
@test J ≈ SMatrix{1,1}([1.])
