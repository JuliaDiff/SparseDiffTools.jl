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


#Sample graph from Gebremedhin AH, Manne F, Pothen A. **What color is your Jacobian? Graph coloring for computing derivatives.**

#=
     (2)
    /  \
   /    \
 (1)----(3)----(4)

=#

gx = SimpleGraph(4)

add_edge!(gx,1,2)
add_edge!(gx,1,3)
add_edge!(gx,2,3)
add_edge!(gx,3,4)

push!(test_graphs, gx)

#begin testing
for i in 1:6
    g = test_graphs[i]

    out_colors1 = SparseDiffTools.color_graph(g,SparseDiffTools.GreedyStar1Color())
    out_colors2 = SparseDiffTools.color_graph(g,SparseDiffTools.GreedyStar2Color())

    #test condition 1
    for v = vertices(g)
        color = out_colors1[v]
        for j in inneighbors(g, v)
            @test out_colors1[j] != color
        end
    end

    #test condition 2
    for j = vertices(g)
        walk = LightGraphs.saw(g, j, 4)
        walk_colors = zeros(Int64, 0)
        if length(walk) >= 4
            for t in walk
                push!(walk_colors, out_colors1[t])
            end
            @test length(unique(walk_colors)) >= 3
        end
    end

    #test condition 1
    for v = vertices(g)
        color = out_colors2[v]
        for j in inneighbors(g, v)
            @test out_colors2[j] != color
        end
    end

    #test condition 2
    for j = vertices(g)
        walk = LightGraphs.saw(g, j, 4)
        walk_colors = zeros(Int64, 0)
        if length(walk) >= 4
            for t in walk
                push!(walk_colors, out_colors2[t])
            end
            @test length(unique(walk_colors)) >= 3
        end
    end

end
