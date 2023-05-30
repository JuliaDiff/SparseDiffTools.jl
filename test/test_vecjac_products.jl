using SparseDiffTools, Zygote
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

a, b = rand(Float32, 2)

A = rand(Float32, N, N)
_f(y, x) = mul!(y, A, x .^ 2)
_f(x) = A * (x .^ 2)

# Define state-dependent functions for operator tests 
include("update_coeffs_testutils.jl")
f = WrapFunc(_f, 1.0f0, 1.0f0)

@test auto_vecjac(f, x, v) ≈ Zygote.jacobian(f, x)[1]' * v
@test auto_vecjac!(zero(x), f, x, v) ≈ auto_vecjac(f, x, v)
@test num_vecjac!(zero(x), f, copy(x), v) ≈ num_vecjac(f, copy(x), v)
@test auto_vecjac(f, x, v) ≈ num_vecjac(f, copy(x), copy(v)) rtol = 1e-2

# Compute Jacobian via Zygote

@info "VecJac AutoZygote"

L = VecJac(f, copy(x), 1.0f0, 1.0f0; autodiff = AutoZygote())

Jx = Zygote.jacobian(f, x)[1]
Jv = Zygote.jacobian(f, v)[1]

@test L * x ≈ Jx' * x
@test L * v ≈ Jx' * v
y=zero(x); @test mul!(y, L, v) ≈ Jx' * v
y=zero(x); @test mul!(y, L, v) ≈ Jx' * v

@test L(x, 1.0f0, 1.0f0) ≈ Jx' * x
y=zero(x); @test L(y, x, 1.0f0, 1.0f0) ≈ Jx' * x
@test L(v, 1.0f0, 1.0f0) ≈ Jv' * v
y=zero(v); @test L(y, v, 1.0f0, 1.0f0) ≈ Jv' * v

update_coefficients!(L, v, 3.0, 4.0)

Jx = Zygote.jacobian(f, x)[1]
Jv = Zygote.jacobian(f, v)[1]

@test L * x ≈ Jv' * x
@test L * v ≈ Jv' * v
y=zero(x); @test mul!(y, L, v) ≈ Jv' * v
y=zero(x); @test mul!(y, L, v) ≈ Jv' * v

@test L(x, 3.0f0, 4.0f0) ≈ Jx' * x
y=zero(x); @test L(y, x, 3.0f0, 4.0f0) ≈ Jx' * x
@test L(v, 3.0f0, 4.0f0) ≈ Jv' * v
y=zero(v); @test L(y, v, 3.0f0, 4.0f0) ≈ Jv' * v

@info "VecJac AutoFiniteDiff"

L = VecJac(f, copy(x), 1.0f0, 1.0f0; autodiff = AutoFiniteDiff())

@test L * x ≈ num_vecjac(f, copy(x), x)
@test L * v ≈ num_vecjac(f, copy(x), v)
y=zero(x); @test mul!(y, L, v) ≈ num_vecjac(f, copy(x), v)

update_coefficients!(L, v, 3.0, 4.0)
@test mul!(y, L, x) ≈ num_vecjac(f, copy(v), x)
_y = copy(y); @test mul!(y, L, x, a, b) ≈ a * num_vecjac(f,copy(v),x) + b * _y

update_coefficients!(f, v, 5.0, 6.0)
@test L(y, v, 5.0, 6.0) ≈ num_vecjac(f, copy(v), v)

# Test that x and v were not mutated
@test x ≈ x0
@test v ≈ v0

@info "Base.resize!"

# Resize test
f2(x) = 2x
f2(y, x) = (copy!(y, x); lmul!(2, y); y)

for M in (100, 400)
    local L = VecJac(f2, copy(x), 1.0f0, 1.0f0; autodiff = AutoZygote())
    resize!(L, M)

    _x = resize!(copy(x), M)
    _u = rand(M)
    J2 = Zygote.jacobian(f2, _x)[1]

    update_coefficients!(L, _x, 1.0f0, 1.0f0)
    @test L * _u ≈ J2' * _u rtol=1e-6
    _v = zeros(M); @test mul!(_v, L, _u) ≈ J2' * _u rtol=1e-6
end

#
