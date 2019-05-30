using SparseArrays
include("custom_graph.jl")

"""
        matrix2graph(SparseMatrix)

A utility function to generate a graph from input
sparse matrix, columns are represented with vertices
and 2 vertices are connected with an edge only if
the two columns are mutually orthogonal.
"""
function matrix2graph(SparseMatrix::SparseMatrixCSC{Int64,Int64})
    dropzeros(SparseMatrix)
    (rows_index, cols_index, val) = findnz(SparseMatrix)

    V = col = size(SparseMatrix, 2)
    row = size(SparseMatrix, 1)

    vertices = zeros(Int64, 0)
    edges = [Vector{Int64}() for _ in 1:V]

    graph = CGraph(vertices, edges)
    for i = 1:V
        push!(vertices, i)
    end

    for i = 1:length(cols_index)
        cur_col = cols_index[i]
        for j = 1:(i-1)
            next_col = cols_index[j]
            if cur_col != next_col
                if row_index[i] == row_index[j]
                    #add edge
                    add_edge!(graph, i, j)
                end
            end
        end
    end
    return graph
end
