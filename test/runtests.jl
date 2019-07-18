using SafeTestsets

@safetestset "Exact coloring via contraction" begin include("test_contraction.jl") end
@safetestset "Greedy distance-1 coloring" begin include("test_greedy_d1.jl") end
@safetestset "Greedy star coloring" begin include("test_greedy_star.jl") end
@safetestset "Matrix to graph conversion" begin include("test_matrix2graph.jl") end
@safetestset "AD using color vector" begin include("test_ad.jl") end
@safetestset "Integration test" begin include("test_integration.jl") end
@safetestset "Special matrices" begin include("test_specialmatrices.jl") end
@safetestset "Jac Vecs and Hes Vecs" begin include("test_jaches_products.jl") end
@safetestset "Program sparsity computation" begin include("program_sparsity/testall.jl") end
