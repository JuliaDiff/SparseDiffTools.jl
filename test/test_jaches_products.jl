using ForwardDiff, LinearAlgebra, Test
const A = rand(300,300)
f(du,u) = mul!(du,A,u)
f(u) = A*u
x = rand(300)
v = rand(300)
du = similar(x)

cache1 = ForwardDiff.Dual{SparseDiffTools.JacVecTag}.(x, v)
cache2 = ForwardDiff.Dual{SparseDiffTools.JacVecTag}.(x, v)
@test num_jacvec!(du, f, x, v) ≈ ForwardDiff.jacobian(f,similar(x),x)*v rtol=1e-6
@test num_jacvec!(du, f, x, v, similar(v), similar(v)) ≈ ForwardDiff.jacobian(f,similar(x),x)*v rtol=1e-6
@test num_jacvec(f, x, v) ≈ ForwardDiff.jacobian(f,similar(x),x)*v rtol=1e-6

@test auto_jacvec!(du, f, x, v) ≈ ForwardDiff.jacobian(f,similar(x),x)*v
@test auto_jacvec!(du, f, x, v, cache1, cache2) ≈ ForwardDiff.jacobian(f,similar(x),x)*v
@test auto_jacvec(f, x, v) ≈ ForwardDiff.jacobian(f,similar(x),x)*v
