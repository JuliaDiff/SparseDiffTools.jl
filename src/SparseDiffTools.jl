module SparseDiffTools

export CGraph, add_vertex!, add_edge!, neighbors, vertices, has_edge, num_edges, has_vertex, rem_edge!, rem_vertex!,
colorGraph, num_vertices, max_degree_vertex, non_neighbors, length_common_neighbor, vertex_degree, contract!, bsc_color

include("custom_graph.jl")
include("contraction_algo.jl")
include("bsc_algo.jl")


end # module
