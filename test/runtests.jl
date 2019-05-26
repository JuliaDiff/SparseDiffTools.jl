using SparseDiffTools
using Test


@testset "Contraction Algorithm" begin include("test_contraction_algo.jl") end
@testset "Greedy D1 Algorithm" begin include("test_greedy_d1.jl") end
