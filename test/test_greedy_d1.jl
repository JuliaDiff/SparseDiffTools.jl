using SparseDiffTools
using VertexSafeGraphs
using LightGraphs
using Random

test_graphs = Array{VSafeGraph, 1}(undef, 0)

for _ in 1:20
    nv = rand(5:25)
    ne = rand(1: nv^2)
    inner = SimpleGraph(nv)
    graph = VSafeGraph(inner)
    for e in 1:ne
        v1 = rand(1:nv)
        v2 = rand(1:nv)
        while v1 == v2
            v2 = rand(1:nv)
        end
        add_edge!(graph, v1, v2)
    end
    push!(test_graphs, copy(graph))
end

for i in 1:20
    g = test_graphs[i]
    coloring = greedy_d1(g)
    out_colors = coloring.colors
    for v = 1:nv(g)
        color = out_colors[v]
        @test color <= coloring.num_colors
        for j in inneighbors(g, v)
            @test out_colors[j] != color
        end
    end
end
