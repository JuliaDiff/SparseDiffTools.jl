include("program_sparsity/common.jl")
include("program_sparsity/basics.jl")

h(y,x, t) = y[:] .= x .+ t
@test sparse(testmeta(h, [1], [2],2)[1]) == sparse([1], [1], true)

@test sparse(testmeta(h, [1], [2],2)[1]) == sparse([1], [1], true)

c(y,x, t) = t > 0 ? y[:] .= x : y[:] .= reverse(x)
@test sparse(testmeta(c, [0,0,0], [1,2,3], 2)[1]) == sparse([1,2,3,1,2,3], [1,2,3,3,2,1], true)
