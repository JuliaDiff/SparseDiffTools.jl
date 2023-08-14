module SparseDiffTools

# QoL/Helper Packages
using Adapt, Compat, Reexport
# Graph Coloring
using Graphs, VertexSafeGraphs
import Graphs: SimpleGraph
# Differentiation
using FiniteDiff, ForwardDiff
@reexport using ADTypes
import ForwardDiff: Dual, jacobian, partials, DEFAULT_CHUNK_THRESHOLD
# Array Packages
using ArrayInterface, SparseArrays
import ArrayInterface: matrix_colors
import StaticArrays
# Others
using SciMLOperators, LinearAlgebra
import DataStructures: DisjointSets, find_root!, union!
import SciMLOperators: update_coefficients, update_coefficients!
import Setfield: @set!
import Tricks: Tricks, static_hasmethod

import PackageExtensionCompat: @require_extensions
function __init__()
    @require_extensions
end

abstract type AbstractAutoDiffVecProd end

include("coloring/high_level.jl")
include("coloring/backtracking_coloring.jl")
include("coloring/contraction_coloring.jl")
include("coloring/greedy_d1_coloring.jl")
include("coloring/acyclic_coloring.jl")
include("coloring/greedy_star1_coloring.jl")
include("coloring/greedy_star2_coloring.jl")
include("coloring/matrix2graph.jl")

include("differentiation/compute_jacobian_ad.jl")
include("differentiation/compute_hessian_ad.jl")
include("differentiation/jaches_products.jl")
include("differentiation/vecjac_products.jl")

Base.@pure __parameterless_type(T) = Base.typename(T).wrapper
parameterless_type(x) = parameterless_type(typeof(x))
parameterless_type(x::Type) = __parameterless_type(x)

function numback_hesvec end
function numback_hesvec! end
function autoback_hesvec end
function autoback_hesvec! end
function auto_vecjac end
function auto_vecjac! end

# Coloring Algorithms
export AcyclicColoring,
    BacktrackingColor, ContractionColor, GreedyD1Color, GreedyStar1Color, GreedyStar2Color
export matrix2graph, matrix_colors
# Sparse Jacobian Computation
export ForwardColorJacCache, forwarddiff_color_jacobian, forwarddiff_color_jacobian!
# Sparse Hessian Computation
export numauto_color_hessian, numauto_color_hessian!, autoauto_color_hessian,
    autoauto_color_hessian!, ForwardAutoColorHesCache, ForwardColorHesCache
# JacVec Products
export auto_jacvec, auto_jacvec!, num_jacvec, num_jacvec!
# VecJac Products
export num_vecjac, num_vecjac!, auto_vecjac, auto_vecjac!
# HesVec Products
export numauto_hesvec,
    numauto_hesvec!, autonum_hesvec, autonum_hesvec!, numback_hesvec, numback_hesvec!
# HesVecGrad Products
export num_hesvecgrad, num_hesvecgrad!, auto_hesvecgrad, auto_hesvecgrad!
# Operators
export JacVec, HesVec, HesVecGrad, VecJac
export update_coefficients, update_coefficients!, value!

end # module
