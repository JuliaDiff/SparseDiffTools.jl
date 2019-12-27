using SparseDiffTools
using LightGraphs
using Test

using Random
Random.seed!(123)

# println("Starting acyclic coloring test...")
#= Test data =#
test_graphs = Vector{SimpleGraph}(undef, 0)
test_graphs_dir = Vector{SimpleDiGraph}(undef, 0)

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

#create directed copies
for g in test_graphs
    dg = SimpleDiGraph(nv(g))
    for e in edges(g)
        src1 = src(e)
        dst1 = dst(e)
        add_edge!(dg, src1, dst1)
        add_edge!(dg, dst1, src1)
    end
    push!(test_graphs_dir, dg)
end


for i in 1:5
    g = test_graphs[i]
    dg = test_graphs_dir[i]

    out_colors = SparseDiffTools.color_graph(g, SparseDiffTools.AcyclicColoring())

    #test condition 1
    for v in vertices(g)
        color = out_colors[v]
        for j in inneighbors(g, v)
            @test out_colors[j] != color
        end
    end
end

for i in 3:4
    g = test_graphs[i]
    dg = test_graphs_dir[i]

    out_colors = SparseDiffTools.color_graph(g, SparseDiffTools.AcyclicColoring())

    #test condition 2
    cycles = simplecycles(dg)
    for c in cycles
        colors = zeros(Int, 0)
        if length(c) > 2
            for v in c
                push!(colors, out_colors[v])
            end
            @test length(unique(colors)) >= 3
        end
    end
    # println("finished testing graph $i")
end

# println("finished testing...")
