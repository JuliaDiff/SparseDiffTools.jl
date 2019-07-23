<<<<<<< HEAD
using SparseDiffTools
using Test


@testset "Exact coloring via contraction" begin include("test_contraction.jl") end
@testset "Exact coloring via backtracking" begin include("test_bsc.jl") end
@testset "Greedy distance-1 coloring" begin include("test_greedy_d1.jl") end
@testset "Greedy star coloring" begin include("test_greedy_star.jl") end
@testset "Matrix to graph conversion" begin include("test_matrix2graph.jl") end
@testset "AD using color vector" begin include("test_ad.jl") end
@testset "Integration test" begin include("test_integration.jl") end
@testset "Special matrices" begin include("test_specialmatrices.jl") end
@testset "Jacobian Vecs and Hessian Vecs" begin include("test_jaches_products.jl") end
@testset "Program sparsity computation" begin include("program_sparsity/testall.jl") end
=======
using SafeTestsets

@time @safetestset "Exact coloring via contraction" begin include("test_contraction.jl") end
@time @safetestset "Greedy distance-1 coloring" begin include("test_greedy_d1.jl") end
@time @safetestset "Greedy star coloring" begin include("test_greedy_star.jl") end
@time @safetestset "Matrix to graph conversion" begin include("test_matrix2graph.jl") end
@time @safetestset "AD using color vector" begin include("test_ad.jl") end
@time @safetestset "Integration test" begin include("test_integration.jl") end
@time @safetestset "Special matrices" begin include("test_specialmatrices.jl") end
@time @safetestset "Jac Vecs and Hes Vecs" begin include("test_jaches_products.jl") end
>>>>>>> 1b115b8d6dd386c46a0474c736e91aae710afd3d
