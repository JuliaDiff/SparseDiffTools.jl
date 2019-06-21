module SparseDiffTools

using SparseArrays, LinearAlgebra, BandedMatrices, BlockBandedMatrices, LightGraphs, VertexSafeGraphs
using BlockBandedMatrices:blocksize,nblocks
using ForwardDiff: Dual, jacobian, partials, DEFAULT_CHUNK_THRESHOLD

export  contract_color,
        greedy_d1,
        matrix2graph,
        matrix_colors,
        forwarddiff_color_jacobian!,
        ForwardColorJacCache

include("coloring/high_level.jl")
include("coloring/contraction_coloring.jl")
include("coloring/greedy_d1_coloring.jl")
include("coloring/matrix2graph.jl")
include("differentiation/compute_jacobian_ad.jl")

end # module
