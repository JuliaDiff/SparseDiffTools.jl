module SparseDiffTools

using SparseArrays, LinearAlgebra, BandedMatrices, BlockBandedMatrices,
      LightGraphs, VertexSafeGraphs, DiffEqDiffTools, ForwardDiff, Zygote,
      SparseArrays
using BlockBandedMatrices:blocksize,nblocks
using ForwardDiff: Dual, jacobian, partials, DEFAULT_CHUNK_THRESHOLD

using Cassette
import Cassette: tag, untag, Tagged, metadata, hasmetadata, istagged, canrecurse
import Cassette: tagged_new_tuple, ContextTagged, BindingMeta, DisableHooks, nametype
import Core: SSAValue

export  contract_color,
        greedy_d1,
        greedy_star1_coloring,
        greedy_star2_coloring,
        matrix2graph,
        matrix_colors,
        forwarddiff_color_jacobian!,
        ForwardColorJacCache,
        auto_jacvec,auto_jacvec!,
        num_jacvec,num_jacvec!,
        num_hesvec,num_hesvec!,
        numauto_hesvec,numauto_hesvec!,
        autonum_hesvec,autonum_hesvec!,
        num_hesvecgrad,num_hesvecgrad!,
        auto_hesvecgrad,auto_hesvecgrad!,
        numback_hesvec,numback_hesvec!,
        autoback_hesvec,autoback_hesvec!,
        JacVec,HesVec,HesVecGrad,
        Sparsity, sparsity!, hsparsity


include("coloring/high_level.jl")
include("coloring/contraction_coloring.jl")
include("coloring/greedy_d1_coloring.jl")
include("coloring/greedy_star1_coloring.jl")
include("coloring/greedy_star2_coloring.jl")
include("coloring/matrix2graph.jl")
include("differentiation/compute_jacobian_ad.jl")
include("differentiation/jaches_products.jl")
include("program_sparsity/program_sparsity.jl")
include("program_sparsity/sparsity_tracker.jl")
include("program_sparsity/path.jl")
include("program_sparsity/take_all_branches.jl")
include("program_sparsity/terms.jl")
include("program_sparsity/linearity.jl")
include("program_sparsity/hessian.jl")
include("program_sparsity/blas.jl")

end # module
