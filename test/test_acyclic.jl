using SparseDiffTools
using LightGraphs
using Test

using Random
Random.seed!(123)

#= Test data =#
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

#=
 Coloring needs to satisfy two conditions:

1. every pair of adjacent vertices receives distinct colors
(a distance-1 coloring)

2. For any vertex v, any color that leads to a two-colored path
involving v and three other vertices  is  impermissible  for  v.
In other words, every path on four vertices uses at least three
colors.
=#


#Sample graph from Gebremedhin AH, Manne F, Pothen A. **New Acyclic and Star Coloring Algorithms with Application to Computing Hessians**

#=
  
(1) ----- (2) ----- (3) ---- (4)
 | \      /          |       |
 |   \   /           |       |
 |     \/            |       |
 (5)---(6) -------- (7) ----(8)
 |      |         /  |       |
 |      |     (11)   |       |
 |      |    /       |       |
 (9) ---(10) ----- (12)-----(13)
  \                 /
   \              /
     ------------
=#

gx = SimpleGraph(13)


add_edge!(gx,1,2)
add_edge!(gx,1,5)
add_edge!(gx,1,6)
add_edge!(gx,2,3)
add_edge!(gx,2,6)
add_edge!(gx,3,4)
add_edge!(gx,3,7)
add_edge!(gx,4,8)
add_edge!(gx,5,6)
add_edge!(gx,5,9)
add_edge!(gx,6,7)
add_edge!(gx,6,10)
add_edge!(gx,7,8)
add_edge!(gx,7,11)
add_edge!(gx,7,12)
add_edge!(gx,8,13)
add_edge!(gx,9,10)
add_edge!(gx,9,12)
add_edge!(gx,10,11)
add_edge!(gx,10,12)
add_edge!(gx,12,13)


push!(test_graphs, gx)

#begin testing
for i in 1:6
    g = test_graphs[i]

    out_colors = SparseDiffTools.color_graph(g,SparseDiffTools.AcyclicColoring())

    #test condition 1
    for v = vertices(g)
        color = out_colors[v]
        for j in inneighbors(g, v)
            @test out_colors[j] != color
        end
    end
end
