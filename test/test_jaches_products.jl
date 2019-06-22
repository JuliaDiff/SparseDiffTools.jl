using ForwardDiff, SparseDiffTools, LinearAlgebra, DiffEqDiffTools,
      IterativeSolvers, Test

const A = rand(300,300)
f(du,u) = mul!(du,A,u)
f(u) = A*u
x = rand(300)
v = rand(300)
du = similar(x)

cache1 = ForwardDiff.Dual{SparseDiffTools.DeivVecTag}.(x, v)
cache2 = ForwardDiff.Dual{SparseDiffTools.DeivVecTag}.(x, v)
@test num_jacvec!(du, f, x, v) ≈ ForwardDiff.jacobian(f,similar(x),x)*v rtol=1e-6
@test num_jacvec!(du, f, x, v, similar(v), similar(v)) ≈ ForwardDiff.jacobian(f,similar(x),x)*v rtol=1e-6
@test num_jacvec(f, x, v) ≈ ForwardDiff.jacobian(f,similar(x),x)*v rtol=1e-6

@test auto_jacvec!(du, f, x, v) ≈ ForwardDiff.jacobian(f,similar(x),x)*v
@test auto_jacvec!(du, f, x, v, cache1, cache2) ≈ ForwardDiff.jacobian(f,similar(x),x)*v
@test auto_jacvec(f, x, v) ≈ ForwardDiff.jacobian(f,similar(x),x)*v

f(u) = sum(u.^2)
@test num_hesvec!(du, f, x, v) ≈ ForwardDiff.hessian(f,x)*v rtol=1e-2
@test num_hesvec!(du, f, x, v, similar(v), similar(v), similar(v)) ≈ ForwardDiff.hessian(f,x)*v rtol=1e-2
@test num_hesvec(f, x, v) ≈ ForwardDiff.hessian(f,x)*v rtol=1e-2

@test numauto_hesvec!(du, f, x, v) ≈ ForwardDiff.hessian(f,x)*v rtol=1e-8
@test numauto_hesvec!(du, f, x, v, ForwardDiff.GradientConfig(f,x), similar(v), similar(v)) ≈ ForwardDiff.hessian(f,x)*v rtol=1e-8
@test numauto_hesvec(f, x, v) ≈ ForwardDiff.hessian(f,x)*v rtol=1e-8

@test autonum_hesvec!(du, f, x, v) ≈ ForwardDiff.hessian(f,x)*v rtol=1e-2
@test autonum_hesvec!(du, f, x, v, similar(v), cache1, cache2) ≈ ForwardDiff.hessian(f,x)*v rtol=1e-2
@test autonum_hesvec(f, x, v) ≈ ForwardDiff.hessian(f,x)*v rtol=1e-8

f(du,u) = mul!(du,A,u)
f(u) = A*u
L = JacVec(f,x)
@test L*x ≈ auto_jacvec(f, x, x)
@test L*v ≈ auto_jacvec(f, x, v)
@test mul!(du,L,v) ≈ auto_jacvec(f, x, v)
L.u .= v
@test mul!(du,L,v) ≈ auto_jacvec(f, v, v)

L = JacVec(f,x,autodiff=false)
@test L*x ≈ num_jacvec(f, x, x)
@test L*v ≈ num_jacvec(f, x, v)
L.u == x
@test mul!(du,L,v) ≈ num_jacvec(f, x, v) rtol=1e-6
L.u .= v
@test mul!(du,L,v) ≈ num_jacvec(f, v, v) rtol=1e-6

### Integration test with IterativeSolvers
out = similar(v)
gmres!(out, L, v)

f(u) = sum(u.^2)
L = HesVec(f,x,autodiff=false)
@test L*x ≈ num_hesvec(f, x, x)
@test L*v ≈ num_hesvec(f, x, v)
@test mul!(du,L,v) ≈ num_hesvec(f, x, v) rtol=1e-2
L.u .= v
@test mul!(du,L,v) ≈ num_hesvec(f, v, v) rtol=1e-2

L = HesVec(f,x)
@test L*x ≈ numauto_hesvec(f, x, x)
@test L*v ≈ numauto_hesvec(f, x, v)
@test mul!(du,L,v) ≈ numauto_hesvec(f, x, v) rtol=1e-8
L.u .= v
@test mul!(du,L,v) ≈ numauto_hesvec(f, v, v) rtol=1e-8

### Integration test with IterativeSolvers
out = similar(v)
gmres!(out, L, v)
