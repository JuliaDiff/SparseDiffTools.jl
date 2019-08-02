"""
        greedy_star2_coloring

    Find a coloring of a given input graph such that
    no two vertices connected by an edge have the same
    color using greedy approach. The number of colors
    used may be equal or greater than the chromatic
    number `χ(G)` of the graph.

    A star coloring is a special type of distance - 1  coloring,
    For a coloring to be called a star coloring, it must satisfy
    two conditions:

    1. every pair of adjacent vertices receives distinct  colors
    (a distance-1 coloring)

    2. For any vertex v, any color that leads to a two-colored path
    involving v and three other vertices  is  impermissible  for  v.
    In other words, every path on four vertices uses at least three
    colors.

    Reference: Gebremedhin AH, Manne F, Pothen A. **What color is your Jacobian? Graph coloring for computing derivatives.** SIAM review. 2005;47(4):629-705.

    TODO: add text explaining the difference between star1 and
    star2
"""
function color_graph(g::LightGraphs.AbstractGraph, :: GreedyStar2Color)
    v = nv(g)
    colorvec = zeros(Int, v)

    forbidden_colors = zeros(Int, v+1)

    for vertex_i = vertices(g)

        for w in inneighbors(g, vertex_i)
            if colorvec[w] != 0
                forbidden_colors[colorvec[w]] = vertex_i
            end

            for x in inneighbors(g, w)
                if colorvec[x] != 0
                    if colorvec[w] == 0
                        forbidden_colors[colorvec[x]] = vertex_i
                    else
                        if colorvec[x] < colorvec[w]
                            forbidden_colors[colorvec[x]] = vertex_i
                        end
                    end
                end
            end
        end

        colorvec[vertex_i] = find_min_color(forbidden_colors, vertex_i)
    end

    colorvec
end
