include("custom_graph.jl")

"""
        matrix2graph(SparseMatrix)

A utility function to generate a graph from input
sparse matrix, columns are represented with vertices
and 2 vertices are connected with an edge only if
the two columns are not mutually orthogonal.
"""
function matrix2graph(SparseMatrix::Array{Int64,2})
    V = col = size(SparseMatrix, 2)
    row = size(SparseMatrix, 1)

    vertices = zeros(Int64,0)
    edges = [Vector{Int64}() for _ in 1:V]
    graph = CGraph(vertices, edges)

    for i = 1:V
        push!(vertices,i)
    end

    for i = 1:V
        for j = 1:(i-1)
            col1 = SparseMatrix[:,i]
            col2 = SparseMatrix[:,j]
            colRes = element_or(col1, col2)
            oneMatrix = ones(Int64, row)
            if oneMatrix' * colRes > 0
                add_edge!(graph, i, j)
            end
        end
    end
    return graph
end

function element_or(col1, col2)
    c3 = zeros(Int64,0)
    for i = 1:length(col1)
        x1 = col1[i]
        y = col2[i]
        if x1 != 0 && y != 0
            res = 1
        else
            res = 0
        end
        push!(c3, res)
    end
    return c3
end
