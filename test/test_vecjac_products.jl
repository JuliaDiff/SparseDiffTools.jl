using SparseDiffTools, Zygote, ForwardDiff
using LinearAlgebra, Test
using StaticArrays

using Random
Random.seed!(123)
N = 300

# Use Float32 since Zygote defaults to Float32
x1 = rand(Float32, N)
x2 = rand(Float32, N)

v = rand(Float32, N)

# Save original values of x and v to make sure they are not ever mutated
_x1 = copy(x1)
_x2 = copy(x2)
_v = copy(v)

a, b = rand(Float32, 2)

A = rand(Float32, N, N)
_f(y, x) = mul!(y, A, x .^ 2)
_f(x) = A * (x .^ 2)
_f2(x, p, t) = _f(x) * p * t
_f2(y, x, p, t) = (_f(y, x); lmul!(p * t, y); y)

# Define state-dependent functions for operator tests
include("update_coeffs_testutils.jl")
f = WrapFunc(_f, 1.0f0, 1.0f0)

@test auto_vecjac(f, x1, v) ≈ Zygote.jacobian(f, x1)[1]' * v
@test auto_vecjac!(zero(x1), f, x1, v) ≈ auto_vecjac(f, x1, v)
@test num_vecjac!(zero(x1), f, copy(x1), v) ≈ num_vecjac(f, copy(x1), v)
@test auto_vecjac(f, x1, v)≈num_vecjac(f, copy(x1), copy(v)) rtol=1e-2

# Compute Jacobian via Zygote

@info "VecJac AutoZygote"

p, t = rand(Float32, 2)
f = WrapFunc(_f, p, t)

L = VecJac(_f2, copy(x1), p, t; autodiff = AutoZygote())
update_coefficients!(L, v, p, t)

update_coefficients!(f, v, p, t)
J1 = Zygote.jacobian(f, x1)[1]
J2 = Zygote.jacobian(f, x2)[1]

# test operator application
@test L * v ≈ J1' * v
@test L(v, p, t) ≈ J1' * v
y = zeros(N);
@test mul!(y, L, v) ≈ J1' * v;
y = zeros(N);
@test L(y, v, p, t) ≈ J1' * v;

# use kwarg VJP_input = x2
@test L(v, p, t; VJP_input = x2) ≈ J2' * v
y = zeros(N);
@test L(y, v, p, t; VJP_input = x2) ≈ J2' * v;

# update_coefficients
p, t = rand(Float32, 2)
L = update_coefficients(L, v, p, t; JVP_input = x2)

update_coefficients!(f, v, p, t)
J1 = Zygote.jacobian(f, x1)[1]
J2 = Zygote.jacobian(f, x2)[1]

# @show p, t
# @show f.p, f.t
# @show L.op.f.p, L.op.f.t

@test L * v ≈ J2' * v
@test L(v, p, t) ≈ J2' * v
y = zeros(N);
@test mul!(y, L, v) ≈ J2' * v;
y = zeros(N);
@test L(y, v, p, t) ≈ J2' * v;

# use kwarg VJP_input = x1
@test L(v, p, t; VJP_input = x1) ≈ J1' * v
y = zeros(N);
@test L(y, v, p, t; VJP_input = x1) ≈ J1' * v;

@info "VecJac AutoFiniteDiff"

p, t = rand(Float32, 2)
f = WrapFunc(_f, p, t)
L = VecJac(_f2, copy(x1), p, t; autodiff = AutoFiniteDiff())
update_coefficients!(L, v, p, t)
update_coefficients!(f, v, p, t)

@test L * v ≈ num_vecjac(f, copy(x1), v)
@test L(v, p, t) ≈ num_vecjac(f, copy(x1), v)
y = zeros(N);
@test mul!(y, L, v) ≈ num_vecjac(f, copy(x1), v);
y = zeros(N);
@test L(y, v, p, t) ≈ num_vecjac(f, copy(x1), v);

# use kwarg VJP_input = x2
@test L(v, p, t; VJP_input = x2) ≈ num_vecjac(f, copy(x2), v)
y = zeros(N);
@test L(y, v, p, t; VJP_input = x2) ≈ num_vecjac(f, copy(x2), v);

# update_coefficients
p, t = rand(Float32, 2)
L = update_coefficients(L, v, p, t; JVP_input = x2)
update_coefficients!(f, v, p, t)

@test L * v ≈ num_vecjac(f, copy(x2), v)
@test L(v, p, t) ≈ num_vecjac(f, copy(x2), v)
y = zeros(N);
@test mul!(y, L, v) ≈ num_vecjac(f, copy(x2), v);
y = zeros(N);
@test L(y, v, p, t) ≈ num_vecjac(f, copy(x2), v);

# use kwarg VJP_input = x2
@test L(v, p, t; VJP_input = x1) ≈ num_vecjac(f, copy(x1), v)
y = zeros(N);
@test L(y, v, p, t; VJP_input = x1) ≈ num_vecjac(f, copy(x1), v);

# Test that x and v were not mutated
@test x1 ≈ _x1
@test x2 ≈ _x2
@test v ≈ v

@info "Base.resize!"

# Resize test
f2(x) = 2x
f2(y, x) = (copy!(y, x); lmul!(2, y); y)

x = rand(Float32, N)
for M in (100, 400)
    local L = VecJac(f2, copy(x); autodiff = AutoZygote())
    resize!(L, M)

    _x = resize!(copy(x), M)
    _u = rand(M)
    local J2 = Zygote.jacobian(f2, _x)[1]

    update_coefficients!(L, _u, nothing, 1.0f0; VJP_input = _x)
    @test L * _u≈J2' * _u rtol=1e-6
    local _v = zeros(M)
    @test mul!(_v, L, _u)≈J2' * _u rtol=1e-6
end

# Test Non-Square Jacobians
f3_oop(x) = vcat(x, x)
function f3_iip(y, x)
    y[1:length(x)] .= x
    y[(length(x) + 1):end] .= x
    return nothing
end

x = rand(Float32, 2)
y = rand(eltype(x), 4)

L = VecJac(f3_oop, copy(x); autodiff = AutoFiniteDiff())
@test size(L) == (length(x), length(y))
@test L * y ≈ num_vecjac(f3_oop, copy(x), y)

L = VecJac(f3_iip, copy(x); autodiff = AutoFiniteDiff(), fu = copy(y))
@test size(L) == (length(x), length(y))
@test mul!(zero(x), L, y) ≈ num_vecjac!(zero(x), f3_iip, copy(x), y)
@test L * y ≈ num_vecjac!(zero(x), f3_iip, copy(x), y)

L = VecJac(f3_oop, copy(x); autodiff = AutoZygote())
@test size(L) == (length(x), length(y))
@test L * y ≈ Zygote.jacobian(f3_oop, copy(x))[1]' * y

@info "Testing StaticArrays"

const A_sa = rand(SMatrix{4, 4, Float32})
_f_sa(x) = A_sa * (x .^ 2)

u = rand(SVector{4, Float32})
v = rand(SVector{4, Float32})

J = ForwardDiff.jacobian(_f_sa, u)
Jᵀv_true = J' * v

@test num_vecjac(_f_sa, u, v) isa SArray
@test num_vecjac(_f_sa, u, v)≈Jᵀv_true atol=1e-3
