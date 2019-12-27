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
    set = DisjointSets{Int}([])

    first_visit_to_tree = Array{Tuple{Int, Int}, 1}()
    first_neighbor = Array{Tuple{Int, Int}, 1}()

    init_array!(first_visit_to_tree, ne(g))
    init_array!(first_neighbor, ne(g))

    forbidden_colors = zeros(Int, nv(g))

    for v in vertices(g)
        for w in outneighbors(g, v)
            if color[w]!=0
                forbidden_colors[color[w]] = v
            end
        end

        for w in outneighbors(g, v)
            if color[w]!=0
                for x in outneighbors(g, w)
                    if color[x]!=0
                        if forbidden_colors[color[x]] != v
                            prevent_cycle!(v, w, x, g, set, first_visit_to_tree, forbidden_colors,color)
                        end
                    end
                end
            end
        end

        color[v] = min_index(forbidden_colors, v)

        for w in outneighbors(g, v)
            if color[w]!=0
                grow_star!(v, w, g, set,first_neighbor,color)
            end
        end

        for w in outneighbors(g, v)
            if color[w]!=0
                for x in outneighbors(g, w)
                    if color[x]!=0 && x!=v
                        if color[x]==color[v]
                            merge_trees!(v,w,x,g,set)
                        end
                    end
                end
            end
        end
    end
    return color
end


"""
    prevent_cycle(v::Integer,
                w::Integer,
                x::Integer,
                g::LightGraphs.AbstractGraph,
                color::AbstractVector{<:Integer},
                forbidden_colors::AbstractVector{<:Integer},
                first_visit_to_tree::Array{Tuple{Integer, Integer}, 1},
                set::DisjointSets{LightGraphs.Edge})

Subroutine to avoid generation of 2-colored cycle due to coloring of vertex v,
which is adjacent to vertices w and x in graph g. Disjoint set is used to store
the induced 2-colored subgraphs/trees where the id of set is a key edge of g
"""
function prevent_cycle!(v::Integer,
                        w::Integer,
                        x::Integer,
                        g::LightGraphs.AbstractGraph,
                        set::DisjointSets{<:Integer},
                        first_visit_to_tree::Array{<:Tuple{Integer,Integer},1},
                        forbidden_colors::AbstractVector{<:Integer},
                        color::AbstractVector{<:Integer})
    e = find(w, x, g, set)
    p, q = first_visit_to_tree[e]

    if p != v
        first_visit_to_tree[e] = (v,w)
    elseif q != w
        forbidden_colors[color[x]] = v
    end
end


"""
    grow_star!(set::DisjointSets{LightGraphs.Edge},
                v::Integer,
                w::Integer,
                g::LightGraphs.AbstractGraph,
                first_neighbor::AbstractVector{<:Tuple{Integer, Integer}},
                color::AbstractVector{<: Integer})

Grow a 2-colored star after assigning a new color to the
previously uncolored vertex v, by comparing it with the adjacent vertex w.
Disjoint set is used to store stars in sets, which are identified through key
edges present in g.
"""
function grow_star!(v::Integer,
                    w::Integer,
                    g::LightGraphs.AbstractGraph,
                    set::DisjointSets{<:Integer},
                    first_neighbor::Array{<: Tuple{Integer,Integer},1},
                    color::AbstractVector{<:Integer})
    make_set!(v,w,g,set)
    p, q = first_neighbor[color[w]]

    if p != v
        first_neighbor[color[w]] = (v,w)
    else
        e1 = find(v,w,g,set)
        e2 = find(p,q,g,set)
        union!(set, e1, e2)
    end
end


"""
        merge_trees!(v::Integer,
                w::Integer,
                x::Integer,
                g::LightGraphs.AbstractGraph,
                set::DisjointSets{LightGraphs.Edge})

Subroutine to merge trees present in the disjoint set which have a
common edge.
"""
function merge_trees!(v::Integer,
                      w::Integer,
                      x::Integer,
                      g::LightGraphs.AbstractGraph,
                      set::DisjointSets{<:Integer})
    e1 = find(v,w,g,set)
    e2 = find(w,x,g,set)
    if e1 != e2
        union!(set, e1, e2)
    end
end


"""
        make_set!(v::Integer,
                  w::Integer,
                  g::LightGraphs.AbstractGraph,
                  set::DisjointSets{<:Integer})

creates a new singleton set in the disjoint set 'set' consisting
of the edge connecting v and w in the graph g
"""
function make_set!(v::Integer,
                   w::Integer,
                   g::LightGraphs.AbstractGraph,
                   set::DisjointSets{<:Integer})
    edge_index = find_edge_index(v,w,g)
    push!(set,edge_index)
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
             set::DisjointSets{<:Integer})

Returns the root of the disjoint set to which the edge connecting vertices w and x
in the graph g belongs to
"""
function find(w::Integer,
              x::Integer,
              g::LightGraphs.AbstractGraph,
              set::DisjointSets{<:Integer})
    edge_index = find_edge_index(w, x, g)
    return find_root(set, edge_index)
end


"""
        find_edge(g::LightGraphs.AbstractGraph, v::Integer, w::Integer)

Returns an integer equivalent to the index of the edge connecting the vertices
v and w in the graph g
"""
function find_edge_index(v::Integer, w::Integer, g::LightGraphs.AbstractGraph)
    pos = 1
    for i in edges(g)

        if (src(i)==v && dst(i)==w) || (src(i)==w && dst(i)==v)
            return pos
        end
        pos = pos + 1
    end
    throw(ArgumentError("$v and $w are not connected in the graph"))
end


"""
        init_array(array::AbstractVector{<:Tuple{Integer, Integer}},
                    n::Integer)

Helper function to initialize the data structures with tuple (0,0)
"""
function init_array!(array::Array{<: Tuple{Integer,Integer},1},
                    n::Integer)
    for i in 1:n
        push!(array,(0,0))
    end
end
