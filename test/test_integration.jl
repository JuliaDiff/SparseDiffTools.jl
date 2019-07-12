using SparseDiffTools, SparseArrays
using LinearAlgebra
using DiffEqDiffTools
using Test

fcalls = 0
function f(dx,x)
  global fcalls += 1
  for i in 2:length(x)-1
    dx[i] = x[i-1] - 2x[i] + x[i+1]
  end
  dx[1] = -2x[1] + x[2]
  dx[end] = x[end-1] - 2x[end]
  nothing
end

function second_derivative_stencil(N)
  A = zeros(N,N)
  for i in 1:N, j in 1:N
      (j-i==-1 || j-i==1) && (A[i,j]=1)
      j-i==0 && (A[i,j]=-2)
  end
  A
end

output = ones(30); input = ones(30)
sparsity_pattern = jacobian_sparsity(f,output,input)
true_jac = Float64.(sparsity_pattern)
colors = matrix_colors(true_jac)
@test colors == repeat(1:3,10)

#Jacobian computed without coloring vector
fcalls = 0
J = DiffEqDiffTools.finite_difference_jacobian(f, rand(30))
@test J ≈ second_derivative_stencil(30)
@test fcalls == 31

#Jacobian computed with coloring vectors
fcalls = 0
_J = 200 .* true_jac
DiffEqDiffTools.finite_difference_jacobian!(_J, f, rand(30), color = colors)
@test fcalls == 4
@test _J ≈ J
