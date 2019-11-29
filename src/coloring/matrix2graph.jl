"""
        matrix2graph(sparse_matrix)

A utility function to generate a graph from input
sparse matrix, columns are represented with vertices
and 2 vertices are connected with an edge only if
the two columns are mutually orthogonal.
"""
function matrix2graph(sparse_matrix::SparseMatrixCSC{<:Number, Int}, partition_by_rows::Bool)
    dropzeros(sparse_matrix)
    (rows_index, cols_index, val) = findnz(sparse_matrix)

    ncols = size(sparse_matrix, 2)
    nrows = size(sparse_matrix, 1)

    num_vtx = partition_by_rows ? nrows : ncols

    inner = SimpleGraph(num_vtx)
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
