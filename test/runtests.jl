using SafeTestsets

@time @safetestset "Exact coloring via contraction" begin include("test_contraction.jl") end
@time @safetestset "Greedy distance-1 coloring" begin include("test_greedy_d1.jl") end
@time @safetestset "Greedy star coloring" begin include("test_greedy_star.jl") end
@time @safetestset "Matrix to graph conversion" begin include("test_matrix2graph.jl") end
@time @safetestset "AD using color vector" begin include("test_ad.jl") end
@time @safetestset "Integration test" begin include("test_integration.jl") end
@time @safetestset "Special matrices" begin include("test_specialmatrices.jl") end
@time @safetestset "Jac Vecs and Hes Vecs" begin include("test_jaches_products.jl") end
