"""
        GreedyD1 Coloring

Find a coloring of a given input graph such that
no two vertices connected by an edge have the same
color using greedy approach. The number of colors
used may be equal or greater than the chromatic
number χ(G) of the graph.
"""
function color_graph(G::VSafeGraph,alg::GreedyD1Color)
    V = nv(G)
    result = zeros(Int64, V)
    result[1] = 1
    available = zeros(Int64, V)
    for i = 2:V
        for j in inneighbors(G, i)
            if result[j] != 0
                available[result[j]] = 1
            end
        end
        for cr = 1:V
            if available[cr] == 0
                result[i] = cr
                break
            end
        end
        available = zeros(Int64, V)
    end
    return result
end
