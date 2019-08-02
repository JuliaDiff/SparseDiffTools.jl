abstract type SparseDiffToolsColoringAlgorithm <: ArrayInterface.ColoringAlgorithm end
struct GreedyD1Color <: SparseDiffToolsColoringAlgorithm end
struct BacktrackingColor <: SparseDiffToolsColoringAlgorithm end
struct ContractionColor <: SparseDiffToolsColoringAlgorithm end
struct GreedyStar1Color <: SparseDiffToolsColoringAlgorithm end
struct GreedyStar2Color <: SparseDiffToolsColoringAlgorithm end

"""
    matrix_colors(A,alg::ColoringAlgorithm = GreedyD1Color())

    Returns the colorvec vector for the matrix A using the chosen coloring
    algorithm. If a known analytical solution exists, that is used instead.
    The coloring defaults to a greedy distance-1 coloring.

"""
function ArrayInterface.matrix_colors(A::AbstractMatrix,alg::SparseDiffToolsColoringAlgorithm = GreedyD1Color(); partition_by_rows::Bool = false)
    _A = A isa SparseMatrixCSC ? A : sparse(A) # Avoid the copy
    A_graph = matrix2graph(_A, partition_by_rows)
    color_graph(A_graph,alg)
end
