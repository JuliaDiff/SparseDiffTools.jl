"""
        greedy_d1_coloring

Find a coloring of a given input graph such that
no two vertices connected by an edge have the same
color using greedy approach. The number of colors
used may be equal or greater than the chromatic
number χ(G) of the graph.
"""
function color_graph(g::VSafeGraph, alg::GreedyD1Color)
    v = nv(g)
    result = zeros(Int, v)
    result[1] = 1
    available = BitArray(undef, v)
    for i = 2:v
        for j in inneighbors(g, i)
            if result[j] != 0
                available[result[j]] = true
            end
        end
        for cr = 1:v
            if available[cr] == false
                result[i] = cr
                break
            end
        end
        fill!(available, false)
    end
    return result
end
