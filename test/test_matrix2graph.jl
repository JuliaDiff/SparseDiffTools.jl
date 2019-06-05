using SparseDiffTools
using SparseArrays
using VertexSafeGraphs
using LightGraphs
using Random

const LG = LightGraphs

n = rand(10:100)
matrices = Array{SparseMatrixCSC, 1}(undef, 0)

for _ in 1:20
    n = rand(10:100)
    matrix = sprand(Int64, n, n, 0.5)
    push!(matrices, copy(matrix))
end

for i in 1:20
    matrix = matrices[i]
    g = matrix2graph(matrix)
    for e in edges(g)
        src = LG.src(e)
        dst = LG.dst(e)
        col1 = abs.(matrix[:, src])
        col2 = abs.(matrix[:, dst])
        pr = col1' * col2
        @test pr != 0
    end
end
