using SparseDiffTools
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

function generate_sparsity_pattern(N::Integer)
    dl = repeat([1.0],N-1)
    du = repeat([1.0],N-1)
    d = repeat([-2.0],N)
    return Tridiagonal(dl,d,du)
end

sparsity_pattern = sparse(generate_sparsity_pattern(30))

_graph = matrix2graph(sparsity_pattern)
coloring_vector = greedy_d1(_graph)
@test coloring_vector == repeat(1:3,10)

#Jacobian computed without coloring vector
J = DiffEqDiffTools.finite_difference_jacobian(f, rand(30))
@test J ≈ second_derivative_stencil(30)
@test fcalls == 31

#Jacobian computed with coloring vectors
fcalls = 0
_J = sparsity_pattern
DiffEqDiffTools.finite_difference_jacobian!(_J, f, rand(30), color = coloring_vector)
@test fcalls == 4
@test _J ≈ J
