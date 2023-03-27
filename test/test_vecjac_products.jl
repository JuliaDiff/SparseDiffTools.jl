using SparseDiffTools, ForwardDiff, FiniteDiff, Zygote, IterativeSolvers
using LinearAlgebra, Test

using Random
Random.seed!(123)
N = 300
const A = rand(N, N)

x = rand(Float32, N)
v = rand(Float32, N)

f(du,u,p,t) = mul!(du, A, u)
f(u,p,t) = A * u

@info "VecJac"

L = VecJac(f, x)
actual_vjp = Zygote.jacobian(x -> f(x, nothing, 0.0), x)[1]' * v
update_coefficients!(L, v, nothing, 0.0)
@test L * v ≈ actual_vjp
L = VecJac(f, x; autodiff = AutoFiniteDiff())
update_coefficients!(L, v, nothing, 0.0)
@test L * v ≈ actual_vjp
#