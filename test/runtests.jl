using SparseDiffTools
using Test


@testset "Exact coloring via contraction" begin include("test_contraction.jl") end
@testset "Greedy distance-1 coloring" begin include("test_greedy_d1.jl") end
@testset "Matrix to graph conversion" begin include("test_matrix2graph.jl") end
@testset "Jacobian sparsity computation" begin include("program_sparsity/testall.jl") end
