#
using SparseDiffTools, ForwardDiff, FiniteDiff, Zygote, IterativeSolvers
using LinearAlgebra, Test

using Random
Random.seed!(123)
N = 300
const A = rand(N, N)
f(y, x) = mul!(y, A, x)
f(x) = A * x
x = rand(N)
v = rand(N)
a, b = rand(2)
dy = similar(x)
g(x) = sum(abs2, x)
function h(x)
    FiniteDiff.finite_difference_gradient(g, x)
end
function h(dy, x)
    FiniteDiff.finite_difference_gradient!(dy, g, x)
end

# JacVecProd

L = JacVecProd(f, x)
@test L * x ≈ auto_jacvec(f, x, x)
@test L * v ≈ auto_jacvec(f, x, v)
@test mul!(dy, L, v) ≈ auto_jacvec(f, x, v)
dy=rand(N);_dy=copy(dy);@test mul!(dy,L,v,a,b) ≈ a*auto_jacvec(f,x,v) + b*_dy
update_coefficients!(L, v, nothing, 0.0)
@test mul!(dy, L, v) ≈ auto_jacvec(f, v, v)
dy=rand(N);_dy=copy(dy);@test mul!(dy,L,v,a,b) ≈ a*auto_jacvec(f,x,v) + b*_dy

L = JacVecProd(f, x, autodiff = false)
@test L * x ≈ num_jacvec(f, x, x)
@test L * v ≈ num_jacvec(f, x, v)
@test mul!(dy, L, v)≈num_jacvec(f, x, v) rtol=1e-6
dy=rand(N);_dy=copy(dy);@test mul!(dy,L,v,a,b) ≈ a*num_jacvec(f,x,v) + b*_dy rtol=1e-6
update_coefficients!(L, v, nothing, 0.0)
@test mul!(dy, L, v)≈num_jacvec(f, v, v) rtol=1e-6
dy=rand(N);_dy=copy(dy);@test mul!(dy,L,v,a,b) ≈ a*num_jacvec(f,x,v) + b*_dy rtol=1e-6

# HesVecProd

x = rand(N)
v = rand(N)
L = HesVecProd(g, x, autodiff = false)
@test L * x ≈ num_hesvec(g, x, x)
@test L * v ≈ num_hesvec(g, x, v)
@test mul!(dy, L, v)≈num_hesvec(g, x, v) rtol=1e-2
dy=rand(N);_dy=copy(dy);@test mul!(dy,L,v,a,b) ≈ a*num_hesvec(g,x,v) + b*_dy rtol=1e-2
update_coefficients!(L, v, nothing, 0.0)
@test mul!(dy, L, v)≈num_hesvec(g, v, v) rtol=1e-2
dy=rand(N);_dy=copy(dy);@test mul!(dy,L,v,a,b) ≈ a*num_hesvec(g,x,v) + b*_dy rtol=1e-2

L = HesVecProd(g, x)
@test L * x ≈ numauto_hesvec(g, x, x)
@test L * v ≈ numauto_hesvec(g, x, v)
@test mul!(dy, L, v)≈numauto_hesvec(g, x, v) rtol=1e-8
dy=rand(N);_dy=copy(dy);@test mul!(dy,L,v,a,b)≈a*numauto_hesvec(g,x,v)+b*_dy rtol=1e-8
update_coefficients!(L, v, nothing, 0.0)
@test mul!(dy, L, v)≈numauto_hesvec(g, v, v) rtol=1e-8
dy=rand(N);_dy=copy(dy);@test mul!(dy,L,v,a,b)≈a*numauto_hesvec(g,x,v)+b*_dy rtol=1e-8

# HesVecGradProd

x = rand(N)
v = rand(N)
L = HesVecGradProd(h, x, autodiff = false)
@test L * x ≈ num_hesvec(g, x, x)
@test L * v ≈ num_hesvec(g, x, v)
@test mul!(dy, L, v)≈num_hesvec(g, x, v) rtol=1e-2
dy=rand(N);_dy=copy(dy);@test mul!(dy,L,v,a,b)≈a*num_hesvec(g,x,v)+b*_dy rtol=1e-2
update_coefficients!(L, v, nothing, 0.0)
@test mul!(dy, L, v)≈num_hesvec(g, v, v) rtol=1e-2
dy=rand(N);_dy=copy(dy);@test mul!(dy,L,v,a,b)≈a*num_hesvec(g,x,v)+b*_dy rtol=1e-2

L = HesVecGradProd(h, x, autodiff = true)
@test L * x ≈ autonum_hesvec(g, x, x)
@test L * v ≈ numauto_hesvec(g, x, v)
@test mul!(dy, L, v)≈numauto_hesvec(g, x, v) rtol=1e-8
dy=rand(N);_dy=copy(dy);@test mul!(dy,L,v,a,b)≈a*numauto_hesvec(g,x,v)+b*_dy rtol=1e-8
update_coefficients!(L, v, nothing, 0.0)
@test mul!(dy, L, v)≈numauto_hesvec(g, v, v) rtol=1e-8
dy=rand(N);_dy=copy(dy);@test mul!(dy,L,v,a,b)≈a*numauto_hesvec(g,x,v)+b*_dy rtol=1e-8

# VecJacProd

f(du,u,p,t) = mul!(du, A, u)
f(u,p,t) = A * u

x = rand(Float32, N)
v = rand(Float32, N)

L = VecJacProd(f, x)
actual_vjp = Zygote.jacobian(x -> f(x, nothing, 0.0), x)[1]' * v
@test L * v ≈ actual_vjp
L = VecJacProd(f, x; autodiff = false)
@test L * v ≈ actual_vjp
#
