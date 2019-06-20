module SparseDiffTools

export  contract_color,
        greedy_d1,
        matrix2graph,
        forwarddiff_color_jacobian!,
        ForwardColorJacCache

include("coloring/contraction_coloring.jl")
include("coloring/greedy_d1_coloring.jl")
include("differentiation/matrix2graph.jl")
include("differentiation/compute_jacobian_ad.jl")

end # module
