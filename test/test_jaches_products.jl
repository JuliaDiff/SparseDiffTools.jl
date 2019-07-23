using SparseDiffTools, ForwardDiff, DiffEqDiffTools, Zygote, IterativeSolvers
using LinearAlgebra, Test

using Random
Random.seed!(123)

const A = rand(300,300)
f(du,u) = mul!(du,A,u)
f(u) = A*u
x = rand(300)
v = rand(300)
du = similar(x)
g(u) = sum(abs2,u)
function h(x)
      DiffEqDiffTools.finite_difference_gradient(g,x)
end
function h(dx,x)
      DiffEqDiffTools.finite_difference_gradient!(dx,g,x)
end

cache1 = ForwardDiff.Dual{SparseDiffTools.DeivVecTag}.(x, v)
cache2 = ForwardDiff.Dual{SparseDiffTools.DeivVecTag}.(x, v)
@test num_jacvec!(du, f, x, v) ≈ ForwardDiff.jacobian(f,similar(x),x)*v rtol=1e-6
@test num_jacvec!(du, f, x, v, similar(v), similar(v)) ≈ ForwardDiff.jacobian(f,similar(x),x)*v rtol=1e-6
@test num_jacvec(f, x, v) ≈ ForwardDiff.jacobian(f,similar(x),x)*v rtol=1e-6

@test auto_jacvec!(du, f, x, v) ≈ ForwardDiff.jacobian(f,similar(x),x)*v
@test auto_jacvec!(du, f, x, v, cache1, cache2) ≈ ForwardDiff.jacobian(f,similar(x),x)*v
@test auto_jacvec(f, x, v) ≈ ForwardDiff.jacobian(f,similar(x),x)*v

@test num_hesvec!(du, g, x, v) ≈ ForwardDiff.hessian(g,x)*v rtol=1e-2
@test num_hesvec!(du, g, x, v, similar(v), similar(v), similar(v)) ≈ ForwardDiff.hessian(g,x)*v rtol=1e-2
@test num_hesvec(g, x, v) ≈ ForwardDiff.hessian(g,x)*v rtol=1e-2

@test numauto_hesvec!(du, g, x, v) ≈ ForwardDiff.hessian(g,x)*v rtol=1e-8
@test numauto_hesvec!(du, g, x, v, ForwardDiff.GradientConfig(g,x), similar(v), similar(v)) ≈ ForwardDiff.hessian(g,x)*v rtol=1e-8
@test numauto_hesvec(g, x, v) ≈ ForwardDiff.hessian(g,x)*v rtol=1e-8

@test autonum_hesvec!(du, g, x, v) ≈ ForwardDiff.hessian(g,x)*v rtol=1e-2
@test autonum_hesvec!(du, g, x, v, similar(v), cache1, cache2) ≈ ForwardDiff.hessian(g,x)*v rtol=1e-2
@test autonum_hesvec(g, x, v) ≈ ForwardDiff.hessian(g,x)*v rtol=1e-8

@test numback_hesvec!(du, g, x, v) ≈ ForwardDiff.hessian(g,x)*v rtol=1e-8
@test numback_hesvec!(du, g, x, v, similar(v), similar(v)) ≈ ForwardDiff.hessian(g,x)*v rtol=1e-8
@test numback_hesvec(g, x, v) ≈ ForwardDiff.hessian(g,x)*v rtol=1e-8

cache3 = ForwardDiff.Dual{Nothing}.(x, v)
cache4 = ForwardDiff.Dual{Nothing}.(x, v)
@test autoback_hesvec!(du, g, x, v) ≈ ForwardDiff.hessian(g,x)*v rtol=1e-8
@test autoback_hesvec!(du, g, x, v, cache3, cache4) ≈ ForwardDiff.hessian(g,x)*v rtol=1e-8
@test autoback_hesvec(g, x, v) ≈ ForwardDiff.hessian(g,x)*v rtol=1e-8

@test num_hesvecgrad!(du, h, x, v) ≈ ForwardDiff.hessian(g,x)*v rtol=1e-2
@test num_hesvecgrad!(du, h, x, v, similar(v), similar(v)) ≈ ForwardDiff.hessian(g,x)*v rtol=1e-2
@test num_hesvecgrad(h, x, v) ≈ ForwardDiff.hessian(g,x)*v rtol=1e-2

@test auto_hesvecgrad!(du, h, x, v) ≈ ForwardDiff.hessian(g,x)*v rtol=1e-2
@test auto_hesvecgrad!(du, h, x, v, cache1, cache2) ≈ ForwardDiff.hessian(g,x)*v rtol=1e-2
@test auto_hesvecgrad(h, x, v) ≈ ForwardDiff.hessian(g,x)*v rtol=1e-2

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

x = rand(300)
v = rand(300)
L = HesVec(g,x,autodiff=false)
@test L*x ≈ num_hesvec(g, x, x)
@test L*v ≈ num_hesvec(g, x, v)
@test mul!(du,L,v) ≈ num_hesvec(g, x, v) rtol=1e-2
L.u .= v
@test mul!(du,L,v) ≈ num_hesvec(g, v, v) rtol=1e-2

L = HesVec(g,x)
@test L*x ≈ numauto_hesvec(g, x, x)
@test L*v ≈ numauto_hesvec(g, x, v)
@test mul!(du,L,v) ≈ numauto_hesvec(g, x, v) rtol=1e-8
L.u .= v
@test mul!(du,L,v) ≈ numauto_hesvec(g, v, v) rtol=1e-8

### Integration test with IterativeSolvers
out = similar(v)
gmres!(out, L, v)

x = rand(300)
v = rand(300)
L = HesVecGrad(h,x,autodiff=false)
@test L*x ≈ num_hesvec(g, x, x)
@test L*v ≈ num_hesvec(g, x, v)
@test mul!(du,L,v) ≈ num_hesvec(g, x, v) rtol=1e-2
L.u .= v
@test mul!(du,L,v) ≈ num_hesvec(g, v, v) rtol=1e-2

L = HesVecGrad(h,x,autodiff=true)
@test L*x ≈ autonum_hesvec(g, x, x)
@test L*v ≈ numauto_hesvec(g, x, v)
@test mul!(du,L,v) ≈ numauto_hesvec(g, x, v) rtol=1e-8
L.u .= v
@test mul!(du,L,v) ≈ numauto_hesvec(g, v, v) rtol=1e-8

### Integration test with IterativeSolvers
out = similar(v)
gmres!(out, L, v)
