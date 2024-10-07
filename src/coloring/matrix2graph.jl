"""
    _cols_by_rows(rows_index,cols_index)

Returns a vector of rows where each row contains
a vector of its column indices.
"""
function _cols_by_rows(rows_index, cols_index)
    nrows = isempty(rows_index) ? 0 : maximum(rows_index)
    cols_by_rows = [eltype(rows_index)[] for _ in 1:nrows]
    for (i, j) in zip(rows_index, cols_index)
        push!(cols_by_rows[i], j)
    end
    return cols_by_rows
end

"""
    _rows_by_cols(rows_index,cols_index)

Returns a vector of columns where each column contains a vector of its row indices.
"""
function _rows_by_cols(rows_index, cols_index)
    return _cols_by_rows(cols_index, rows_index)
end

"""
    matrix2graph(sparse_matrix, [partition_by_rows::Bool=true])

A utility function to generate a graph from input sparse matrix, columns are represented
with vertices and 2 vertices are connected with an edge only if the two columns are mutually
orthogonal.

Note that the sparsity pattern is defined by structural nonzeroes, ie includes explicitly
stored zeros.
"""
function matrix2graph(sparse_matrix::AbstractSparseMatrix{<:Number},
        partition_by_rows::Bool = true)
    (rows_index, cols_index, _) = findnz(sparse_matrix)

    ncols = size(sparse_matrix, 2)
    nrows = size(sparse_matrix, 1)

    num_vtx = partition_by_rows ? nrows : ncols

    inner = SimpleGraph{promote_type(eltype(rows_index), eltype(cols_index))}(num_vtx)

    if partition_by_rows
        rows_by_cols = _rows_by_cols(rows_index, cols_index)
        for (cur_row, cur_col) in zip(rows_index, cols_index)
            if !isempty(rows_by_cols[cur_col])
                for next_row in rows_by_cols[cur_col]
                    if next_row < cur_row
                        add_edge!(inner, cur_row, next_row)
                    end
                end
            end
        end
    else
        cols_by_rows = _cols_by_rows(rows_index, cols_index)
        for (cur_row, cur_col) in zip(rows_index, cols_index)
            if !isempty(cols_by_rows[cur_row])
                for next_col in cols_by_rows[cur_row]
                    if next_col < cur_col
                        add_edge!(inner, cur_col, next_col)
                    end
                end
            end
        end
    end
    return VSafeGraph(inner)
end
