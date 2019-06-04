module SparseDiffTools

export  contract_color,
        greedy_d1,
        matrix2graph

include("contraction_coloring.jl")
include("greedy_d1_coloring.jl")
include("matrix2graph.jl")


end # module
