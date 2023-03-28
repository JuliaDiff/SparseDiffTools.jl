using SparseDiffTools, ForwardDiff, FiniteDiff, Zygote, IterativeSolvers
using LinearAlgebra, Test

using Random
Random.seed!(123)
N = 300

# Use Float32 since Zygote defaults to Float32
x = rand(Float32, N)
v = rand(Float32, N)

# Save original values of x and v to make sure they are not ever mutated
x0 = copy(x)
v0 = copy(v)

a, b = rand(2)
dy = similar(x)

A = rand(Float32, N, N)
_f(du,u) = mul!(du, A, u)
_f(u) = A * u

# Define state-dependent functions for operator tests 
include("update_coeffs_testutils.jl")
f = WrapFunc(_f, 1.0f0, 1.0f0)

# Compute Jacobian via Zygote

@info "VecJac"

L = VecJac(f, copy(x), 1.0f0, 1.0f0; autodiff = AutoZygote())
update_coefficients!(f, x, 1.0, 1.0)
actual_jac = Zygote.jacobian(f, x)[1]
@test L * x ≈ actual_jac' * x 
@test L * v ≈ actual_jac' * v 
@test mul!(dy, L, v) ≈ actual_jac' * v 
update_coefficients!(L, v, 3.0, 4.0)
update_coefficients!(f, v, 3.0, 4.0)
actual_jac = Zygote.jacobian(f, v)[1]
@test mul!(dy, L, x) ≈ actual_jac' * x
_dy=copy(dy); @test mul!(dy,L,x,a,b) ≈ a*actual_jac'*x + b*_dy
update_coefficients!(f, v, 5.0, 6.0)
actual_jac = Zygote.jacobian(f, v)[1]
@test L(dy, v, 5.0, 6.0) ≈ actual_jac' * v

L = VecJac(f, copy(x), 1.0f0, 1.0f0; autodiff = AutoFiniteDiff())
update_coefficients!(f, x, 1.0, 1.0)
actual_jac = Zygote.jacobian(f, x)[1]
@test L * x ≈ actual_jac' * x 
@test L * v ≈ actual_jac' * v 
@test mul!(dy, L, v) ≈ actual_jac' * v 
update_coefficients!(L, v, 3.0, 4.0)
update_coefficients!(f, v, 3.0, 4.0)
actual_jac = Zygote.jacobian(f, v)[1]
@test mul!(dy, L, x) ≈ actual_jac' * x
_dy=copy(dy); @test mul!(dy,L,x,a,b) ≈ a*actual_jac'*x + b*_dy
update_coefficients!(f, v, 5.0, 6.0)
actual_jac = Zygote.jacobian(f, v)[1]
@test L(dy, v, 5.0, 6.0) ≈ actual_jac' * v

# Test that x and v were not mutated
@test x ≈ x0
@test v ≈ v0
#
