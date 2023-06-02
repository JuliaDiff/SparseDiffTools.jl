using Pkg
using SafeTestsets

const GROUP = get(ENV, "GROUP", "All")
const is_APPVEYOR = (Sys.iswindows() && haskey(ENV, "APPVEYOR"))
const is_TRAVIS = haskey(ENV, "TRAVIS")

function activate_gpu_env()
    Pkg.activate("gpu")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    Pkg.instantiate()
end

if GROUP == "Core" || GROUP == "All"
    @time @safetestset "Exact coloring via contraction" include("test_contraction.jl")
    @time @safetestset "Greedy distance-1 coloring" include("test_greedy_d1.jl")
    @time @safetestset "Greedy star coloring" include("test_greedy_star.jl")
    @time @safetestset "Acyclic coloring" include("test_acyclic.jl")
    @time @safetestset "Matrix to graph conversion" include("test_matrix2graph.jl")
    @time @safetestset "Hessian colorvecs" include("test_sparse_hessian.jl")
    @time @safetestset "Integration test" include("test_integration.jl")
    @time @safetestset "Special matrices" include("test_specialmatrices.jl")
    @time @safetestset "AD using colorvec vector" include("test_ad.jl")
end

if GROUP == "InterfaceI" || GROUP == "All"
    @time @safetestset "Jac Vecs and Hes Vecs" include("test_jaches_products.jl")
    @time @safetestset "Vec Jac Products" include("test_vecjac_products.jl")
end

if GROUP == "GPU"
    activate_gpu_env()
    @time @safetestset "GPU AD" include("test_gpu_ad.jl")
end
