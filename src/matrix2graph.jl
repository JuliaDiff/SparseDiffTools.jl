using SparseArrays
using LightGraphs
using VertexSafeGraphs

"""
        matrix2graph(SparseMatrix)

A utility function to generate a graph from input
sparse matrix, columns are represented with vertices
and 2 vertices are connected with an edge only if
the two columns are mutually orthogonal.
"""
function matrix2graph(SparseMatrix::SparseMatrixCSC{T,Int}) where T<:Number
    dropzeros(SparseMatrix)
    (rows_index, cols_index, val) = findnz(SparseMatrix)

    V = cols = size(SparseMatrix, 2)
    rows = size(SparseMatrix, 1)

    inner = SimpleGraph(V)
    graph = VSafeGraph(inner)

    for i = 1:length(cols_index)
        cur_col = cols_index[i]
        for j = 1:(i-1)
            next_col = cols_index[j]
            if cur_col != next_col
                if rows_index[i] == rows_index[j]
                    add_edge!(graph, cur_col, next_col)
                end
            end
        end
    end
    return graph
end
