"""
        matrix2graph(SparseMatrix)

A utility function to generate a graph from input
sparse matrix, columns are represented with vertices
and 2 vertices are connected with an edge only if
the two columns are mutually orthogonal.
"""
function matrix2graph(SparseMatrix::SparseMatrixCSC{T,Int}, partition_by_rows::Bool) where T<:Number
    dropzeros(SparseMatrix)
    (rows_index, cols_index, val) = findnz(SparseMatrix)

    cols = size(SparseMatrix, 2)
    rows = size(SparseMatrix, 1)

    partition_by_rows ? V = rows : V = cols

    inner = SimpleGraph(V)
    graph = VSafeGraph(inner)

    if partition_by_rows
        for i = 1:length(rows_index)
            cur_row = rows_index[i]
            for j = 1:(i-1)
                next_row = rows_index[j]
                if cur_row != next_row
                    if cols_index[i] == cols_index[j]
                        add_edge!(graph, cur_row, next_row)
                    end
                end
            end
        end
    else
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
    end
    return graph
end
