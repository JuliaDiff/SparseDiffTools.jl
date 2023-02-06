#
using SparseDiffTools, ForwardDiff, FiniteDiff, Zygote, IterativeSolvers
using LinearAlgebra, Test

using Random
Random.seed!(123)

const A = rand(300, 300)
f(y, x) = mul!(y, A, x)
f(x) = A * x
x = rand(300)
v = rand(300)
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
update_coefficients!(L, v, nothing, 0.0)
@test mul!(dy, L, v) ≈ auto_jacvec(f, v, v)

L = JacVecProd(f, x, autodiff = false)
@test L * x ≈ num_jacvec(f, x, x)
@test L * v ≈ num_jacvec(f, x, v)
update_coefficients!(L, x, nothing, 0.0)
@test mul!(dy, L, v)≈num_jacvec(f, x, v) rtol=1e-6
update_coefficients!(L, v, nothing, 0.0)
@test mul!(dy, L, v)≈num_jacvec(f, v, v) rtol=1e-6

# HesVecProd

x = rand(300)
v = rand(300)
L = HesVecProd(g, x, autodiff = false)
@test L * x ≈ num_hesvec(g, x, x)
@test L * v ≈ num_hesvec(g, x, v)
@test mul!(dy, L, v)≈num_hesvec(g, x, v) rtol=1e-2
update_coefficients!(L, v, nothing, 0.0)
@test mul!(dy, L, v)≈num_hesvec(g, v, v) rtol=1e-2

L = HesVecProd(g, x)
@test L * x ≈ numauto_hesvec(g, x, x)
@test L * v ≈ numauto_hesvec(g, x, v)
@test mul!(dy, L, v)≈numauto_hesvec(g, x, v) rtol=1e-8
update_coefficients!(L, v, nothing, 0.0)
@test mul!(dy, L, v)≈numauto_hesvec(g, v, v) rtol=1e-8

# HesVecGradProd

x = rand(300)
v = rand(300)
L = HesVecGradProd(h, x, autodiff = false)
@test L * x ≈ num_hesvec(g, x, x)
@test L * v ≈ num_hesvec(g, x, v)
@test mul!(dy, L, v)≈num_hesvec(g, x, v) rtol=1e-2
update_coefficients!(L, v, nothing, 0.0)
@test mul!(dy, L, v)≈num_hesvec(g, v, v) rtol=1e-2

L = HesVecGradProd(h, x, autodiff = true)
@test L * x ≈ autonum_hesvec(g, x, x)
@test L * v ≈ numauto_hesvec(g, x, v)
@test mul!(dy, L, v)≈numauto_hesvec(g, x, v) rtol=1e-8
update_coefficients!(L, v, nothing, 0.0)
@test mul!(dy, L, v)≈numauto_hesvec(g, v, v) rtol=1e-8

# VecJacProd
#
