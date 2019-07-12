include("common.jl")

@testset "Basics" begin include("jacobian.jl") end
@testset "Hessian sparsity" begin include("hessian.jl") end
@testset "Paraboloid example" begin include("paraboloid.jl") end

@testset "Exploration" begin include("ifsandbuts.jl") end
