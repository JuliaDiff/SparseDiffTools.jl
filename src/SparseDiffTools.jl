module SparseDiffTools

export  contract_color,
        greedy_d1,
        matrix2graph,
        Coloring

"""
        Coloring{T}

Stores the number of colors used in a graph
and the mapping of colors to vertices
"""
struct Coloring{T <: Integer}
        num_colors::T
        colors::Vector{T}
end


include("contraction_coloring.jl")
include("greedy_d1_coloring.jl")
include("matrix2graph.jl")


end # module
