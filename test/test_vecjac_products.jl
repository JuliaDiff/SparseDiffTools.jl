using SparseDiffTools, ForwardDiff, FiniteDiff, Zygote, IterativeSolvers
using LinearAlgebra, Test

using Random
Random.seed!(123)
N = 300
const A = rand(N, N)

x = rand(Float32, N)
v = rand(Float32, N)

_f(du,u) = mul!(du, A, u)
_f(u) = A * u

# Define state-dependent functions for operator tests 
include("update_coeffs_testutils.jl")
f = WrapFunc(_f, 1.0, 1.0)

@info "VecJac"

L = VecJac(f, x, 1.0, 1.0)
update_coefficients!(L, v, 3.0, 4.0)
update_coefficients!(f, v, 3.0, 4.0)
actual_vjp = Zygote.jacobian(f, x)[1]' * v
@test L * v ≈ actual_vjp
update_coefficients!(f, v, 5.0, 6.0)
actual_vjp2 = Zygote.jacobian(f, x)[1]' * v
@test L(copy(v), v, 5.0, 6.0) ≈ actual_vjp2

L = VecJac(f, x, 1.0, 1.0; autodiff = AutoFiniteDiff())
update_coefficients!(L, v, 3.0, 4.0)
update_coefficients!(f, v, 3.0, 4.0)
@test L * v ≈ actual_vjp
update_coefficients!(f, v, 5.0, 6.0)
@test L(copy(v), v, 5.0, 6.0) ≈ actual_vjp2
#
