using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = ( Sys.iswindows() && haskey(ENV,"APPVEYOR") )
const is_TRAVIS = haskey(ENV,"TRAVIS")

if GROUP == "All"
    @time @safetestset "Exact coloring via contraction" begin include("test_contraction.jl") end
    @time @safetestset "Greedy distance-1 coloring" begin include("test_greedy_d1.jl") end
    @time @safetestset "Greedy star coloring" begin include("test_greedy_star.jl") end
    @time @safetestset "Matrix to graph conversion" begin include("test_matrix2graph.jl") end
    @time @safetestset "AD using colorvec vector" begin include("test_ad.jl") end
    @time @safetestset "Integration test" begin include("test_integration.jl") end
    @time @safetestset "Special matrices" begin include("test_specialmatrices.jl") end
    @time @safetestset "Jac Vecs and Hes Vecs" begin include("test_jaches_products.jl") end
end

if GROUP == "GPU"
    @time @safetestset "GPU AD" begin include("test_gpu_ad.jl") end
end
