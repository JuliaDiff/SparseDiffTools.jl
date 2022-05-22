using SparseDiffTools
using SparseArrays
using VertexSafeGraphs
using Graphs
using Random

n = rand(10:100)
matrices = Array{SparseMatrixCSC, 1}(undef, 0)

for _ in 1:20
    n = rand(10:100)
    matrix = sprand(Int64, n, n, 0.5)
    push!(matrices, copy(matrix))
end

for i in 1:20
    matrix = matrices[i]
    g = matrix2graph(matrix, false)
    for e in edges(g)
        src = Graphs.src(e)
        dst = Graphs.dst(e)
        col1 = abs.(matrix[:, src])
        col2 = abs.(matrix[:, dst])
        pr = col1' * col2
        @test pr != 0
    end
end

for i in 1:20
    matrix = matrices[i]
    g = matrix2graph(matrix, true)
    for e in edges(g)
        src = Graphs.src(e)
        dst = Graphs.dst(e)
        row1 = abs.(matrix[src, :])
        row2 = abs.(matrix[dst, :])
        pr = row1' * row2
        @test pr != 0
    end
end

@info "stored zeros"
for i in 1:20  
    matrix = matrices[i]
    g = matrix2graph(matrix)
    # recalculate graph with stored zeros
    matrix_sz = copy(matrix)
    fill!(matrix_sz, 0.0)
    g_sz = matrix2graph(matrix_sz)
    # check that graphs are the same 
    @test nv(g) == nv(g_sz)
    @test ne(g) == ne(g_sz)
    for e in edges(g)
        @test has_edge(g_sz, e)
    end
    for e in edges(g_sz)
        @test has_edge(g, e)
    end    
end
