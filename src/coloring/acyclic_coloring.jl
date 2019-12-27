"""
        color_graph(g::LightGraphs.AbstractGraphs, ::AcyclicColoring)

Returns a coloring vector following the acyclic coloring rules (1) the coloring
corresponds to a distance-1 coloring, and (2) vertices in every cycle of the
graph are assigned at least three distinct colors. This variant of coloring is
called acyclic since every subgraph induced by vertices assigned any two colors
is a collection of trees—and hence is acyclic.

Reference: Gebremedhin AH, Manne F, Pothen A. **New Acyclic and Star Coloring Algorithms with Application to Computing Hessians**
"""
function color_graph(g::LightGraphs.AbstractGraph, ::AcyclicColoring)
    color = zeros(Int, nv(g))
    two_colored_forest = DisjointSets{Int}(())

    first_visit_to_tree = fill((0,0), ne(g))
    first_neighbor = fill((0,0), ne(g))

    forbidden_colors = zeros(Int, nv(g))

    for v in vertices(g)
        for w in outneighbors(g, v)
            if color[w] != 0
                forbidden_colors[color[w]] = v
            end
        end

        for w in outneighbors(g, v)
            if color[w] != 0
                for x in outneighbors(g, w)
                    if color[x] != 0
                        if forbidden_colors[color[x]] != v
                            prevent_cycle!(first_visit_to_tree,forbidden_colors,v, w, x, g, two_colored_forest,color)
                        end
                    end
                end
            end
        end

        color[v] = min_index(forbidden_colors, v)

        for w in outneighbors(g, v)
            if color[w] != 0
                grow_star!(two_colored_forest,first_neighbor,v, w, g, color)
            end
        end

        for w in outneighbors(g, v)
            if color[w] != 0
                for x in outneighbors(g, w)
                    if color[x] != 0 && x != v
                        if color[x] == color[v]
                            merge_trees!(two_colored_forest,v,w,x,g)
                        end
                    end
                end
            end
        end
    end
    return color
end


"""
        prevent_cycle!(first_visit_to_tree::AbstractVector{<:Tuple{Integer,Integer}},
                        forbidden_colors::AbstractVector{<:Integer},
                        v::Integer,
                        w::Integer,
                        x::Integer,
                        g::LightGraphs.AbstractGraph,
                        two_colored_forest::DisjointSets{<:Integer},
                        color::AbstractVector{<:Integer})

Subroutine to avoid generation of 2-colored cycle due to coloring of vertex v,
which is adjacent to vertices w and x in graph g. Disjoint set is used to store
the induced 2-colored subgraphs/trees where the id of set is an integer
representing an edge of graph 'g'
"""
function prevent_cycle!(first_visit_to_tree::AbstractVector{<:Tuple{Integer,Integer}},
                        forbidden_colors::AbstractVector{<:Integer},
                        v::Integer,
                        w::Integer,
                        x::Integer,
                        g::LightGraphs.AbstractGraph,
                        two_colored_forest::DisjointSets{<:Integer},
                        color::AbstractVector{<:Integer})
    e = find(w, x, g, two_colored_forest)
    p, q = first_visit_to_tree[e]

    if p != v
        first_visit_to_tree[e] = (v,w)
    elseif q != w
        forbidden_colors[color[x]] = v
    end
end


"""
        grow_star!(two_colored_forest::DisjointSets{<:Integer},
                    first_neighbor::AbstractVector{<: Tuple{Integer,Integer}},
                    v::Integer,
                    w::Integer,
                    g::LightGraphs.AbstractGraph,
                    color::AbstractVector{<:Integer})

Grow a 2-colored star after assigning a new color to the
previously uncolored vertex v, by comparing it with the adjacent vertex w.
Disjoint set is used to store stars in sets, which are identified through key
edges present in g.
"""
function grow_star!(two_colored_forest::DisjointSets{<:Integer},
                    first_neighbor::AbstractVector{<: Tuple{Integer,Integer}},
                    v::Integer,
                    w::Integer,
                    g::LightGraphs.AbstractGraph,
                    color::AbstractVector{<:Integer})
    insert_new_tree!(two_colored_forest,v,w,g)
    p, q = first_neighbor[color[w]]

    if p != v
        first_neighbor[color[w]] = (v,w)
    else
        e1 = find(v,w,g,two_colored_forest)
        e2 = find(p,q,g,two_colored_forest)
        union!(two_colored_forest, e1, e2)
    end
end


"""
        merge_trees!(two_colored_forest::DisjointSets{<:Integer},
                      v::Integer,
                      w::Integer,
                      x::Integer,
                      g::LightGraphs.AbstractGraph)

Subroutine to merge trees present in the disjoint set which have a
common edge.
"""
function merge_trees!(two_colored_forest::DisjointSets{<:Integer},
                      v::Integer,
                      w::Integer,
                      x::Integer,
                      g::LightGraphs.AbstractGraph)
    e1 = find(v,w,g,two_colored_forest)
    e2 = find(w,x,g,two_colored_forest)
    if e1 != e2
        union!(two_colored_forest, e1, e2)
    end
end


"""
        insert_new_tree!(two_colored_forest::DisjointSets{<:Integer},
                          v::Integer,
                          w::Integer,
                          g::LightGraphs.AbstractGraph)

creates a new singleton set in the disjoint set 'two_colored_forest' consisting
of the edge connecting v and w in the graph g
"""
function insert_new_tree!(two_colored_forest::DisjointSets{<:Integer},
                          v::Integer,
                          w::Integer,
                          g::LightGraphs.AbstractGraph)
    edge_index = find_edge_index(v,w,g)
    push!(two_colored_forest,edge_index)
end


"""
        min_index(forbidden_colors::AbstractVector{<:Integer}, v::Integer)

Returns min{i > 0 such that forbidden_colors[i] != v}
"""
function min_index(forbidden_colors::AbstractVector{<:Integer}, v::Integer)
    return findfirst(!isequal(v), forbidden_colors)
end


"""
        find(w::Integer,
             x::Integer,
             g::LightGraphs.AbstractGraph,
             two_colored_forest::DisjointSets{<:Integer})

Returns the root of the disjoint set to which the edge connecting vertices w and x
in the graph g belongs to
"""
function find(w::Integer,
              x::Integer,
              g::LightGraphs.AbstractGraph,
              two_colored_forest::DisjointSets{<:Integer})
    edge_index = find_edge_index(w, x, g)
    return find_root(two_colored_forest, edge_index)
end


"""
        find_edge(g::LightGraphs.AbstractGraph, v::Integer, w::Integer)

Returns an integer equivalent to the index of the edge connecting the vertices
v and w in the graph g
"""
function find_edge_index(v::Integer, w::Integer, g::LightGraphs.AbstractGraph)
    pos = 1
    for i in edges(g)

        if (src(i) == v && dst(i) == w) || (src(i) == w && dst(i) == v)
            return pos
        end
        pos = pos + 1
    end
    throw(ArgumentError("$v and $w are not connected in the graph"))
end
