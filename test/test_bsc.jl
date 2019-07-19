using SparseDiffTools
using LightGraphs
using Random
#=
            Graph g0
vertex nuumber followed by color in parentheses

            6(3)
           /    \
          /      \
         /        \
       5(1)--------3(2)
       |  \        |
       |   \       |
       |    \      |
       |   1(2)----2(1)
       |   /
       |  /
      4(3)
=#

g0 = SimpleGraph(6)
add_edge!(g0, 1,2)
add_edge!(g0, 1,4)
add_edge!(g0, 1,5)
add_edge!(g0, 3,2)
add_edge!(g0, 3,5)
add_edge!(g0, 3,6)
add_edge!(g0, 4,5)
add_edge!(g0, 5,6)


#=
        Graph g1

          1
          |
          |
     5----2----3
         / \
        /   \
      6      4

=#
g1 = SimpleGraph(6)
add_edge!(g1, 2,1)
add_edge!(g1, 3,2)
add_edge!(g1, 4,2)
add_edge!(g1, 5,2)
add_edge!(g1, 6,2)


#=
        Graph g2

            1------2
          / |    / |
         /  |   /  |
        /   | /    |
       3----4------5

=#
g2 = SimpleGraph(5)
add_edge!(g2, 1,2)
add_edge!(g2, 1,3)
add_edge!(g2, 1,4)
add_edge!(g2, 4,2)
add_edge!(g2, 5,2)
add_edge!(g2, 3,4)
add_edge!(g2, 4,5)

#test custom graphs first
coloring0 = SparseDiffTools.color_graph(g0, SparseDiffTools.BacktrackingColor())
coloring1 = SparseDiffTools.color_graph(g1, SparseDiffTools.BacktrackingColor())
coloring2 = SparseDiffTools.color_graph(g2, SparseDiffTools.BacktrackingColor())

for v = 1:nv(g0)
    color = coloring0[v]
    for j in inneighbors(g0, v)
        if coloring0[j] == color
             @test false
        end
    end
end

for v = 1:nv(g1)
    color = coloring1[v]
    for j in inneighbors(g1, v)
        if coloring1[j] == color
             @test false
        end
    end
end

for v = 1:nv(g2)
    color = coloring2[v]
    for j in inneighbors(g2, v)
        if coloring2[j] == color
             @test false
        end
    end
end

test_graphs = Array{SimpleGraph, 1}(undef, 0)
for _ in 1:5
    nv = rand(5:20)
    ne = rand(1:100)
    graph = SimpleGraph(nv)
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


for i in 1:5
    g = test_graphs[i]
    out_colors = SparseDiffTools.color_graph(g,SparseDiffTools.BacktrackingColor())

    for v = vertices(g)
        color = out_colors[v]
        for j in inneighbors(g, v)
            @test out_colors[j] != color
        end
    end
end
