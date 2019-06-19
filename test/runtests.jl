using SparseDiffTools
using Test


@testset "Exact coloring via contraction" begin include("test_contraction.jl") end
@testset "Greedy distance-1 coloring" begin include("test_greedy_d1.jl") end
@testset "Matrix to graph conversion" begin include("test_matrix2graph.jl") end
@testset "AD using color vector" begin include("test_ad.jl") end
@testset "Integration test" begin include("test_integration.jl") end
