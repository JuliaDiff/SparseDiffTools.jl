using LightGraphs
using DataStructures

"""
        color_graph(g::LightGraphs.AbstractGraphs, ::AcyclicColoring)

Returns a coloring vector following the acyclic coloring rules (1) the coloring 
corresponds to a distance-1 coloring, and (2) vertices in every cycle of the 
graph are assigned at least three distinct colors. This variant of coloring is 
called acyclic since every subgraph induced by vertices assigned any two colors
is a collection of trees—and hence is acyclic.
"""
function color_graph(g::LightGraphs.AbstractGraph, ::AcyclicColoring)
    
    color = zeros(Int, nv(g))
    forbiddenColors = zeros(Int, nv(g))

    set = DisjointSets{LightGraphs.Edge}([])

    firstVisitToTree = Array{Tuple{Int, Int}, 1}(undef, ne(g))
    firstNeighbor = Array{Tuple{Int, Int}, 1}(undef, nv(g))

    for v in vertices(g)
        #enforces the first condition of acyclic coloring
        for w in outneighbors(g, v)
            if color[w] != 0
                forbiddenColors[color[w]] = v
            end
        end
        #enforces the second condition of acyclic coloring
        for w in outneighbors(g, v)
            if color[w] != 0 #colored neighbor
                for x in outneighbors(g, w)
                    if color[x] != 0 #colored x
                        if forbiddenColors[color[x]] != v
                            prevent_cycle(v, w, x, g, color, forbiddenColors, firstVisitToTree, set)
                        end
                    end
                end
            end
        end

        color[v] = min_index(forbiddenColors, v)

        #grow star for every edge connecting colored vertices v and w
        for w in outneighbors(g, v)
            if color[w] != 0
                grow_star(v, w, g, firstNeighbor, set)
            end
        end

        #merge the newly formed stars into existing trees if possible
        for w in outneighbors(g, v)
            if color[w] != 0
                for x in outneighbors(g, w)
                    if color[x] != 0 && x != v
                        if color[x] == color[v]
                            merge_trees(v, w, x, g, set)
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
                    forbiddenColors::AbstractVector{<:Integer},
                    firstVisitToTree::Array{Tuple{Integer, Integer}, 1},
                    set::DisjointSets{LightGraphs.Edge})

Subroutine to avoid generation of 2-colored cycle due to coloring of vertex v,
which is adjacent to vertices w and x in graph g. Disjoint set is used to store
the induced 2-colored subgraphs/trees where the id of set is a key edge of g
"""
function prevent_cycle(v::Integer,
                       w::Integer, 
                       x::Integer,
                       g::LightGraphs.AbstractGraph,
                       color::AbstractVector{<:Integer}, 
                       forbiddenColors::AbstractVector{<:Integer},
                       firstVisitToTree::Array{Tuple{Integer, Integer}, 1},
                       set::DisjointSets{LightGraphs.Edge})
    
    edge = find_edge(g, w, x)
    e = find_root(set, edge)
    p, q = firstVisitToTree[edge_index(g, e)]
    if p != v
        firstVisitToTree[edge_index(g, e)] = (v, w)
    else if q != w
        forbiddenColors[color[x]] = v
    end
end

"""
        min_index(forbiddenColors::AbstractVector{<:Integer}, v::Integer)

Returns min{i > 0 such that forbiddenColors[i] != v}
"""            
function min_index(forbiddenColors::AbstractVector{<:Integer},
                   v::Integer)
    for i = 1:length(forbiddenColors)
        if forbiddenColors[i] != v
            return i
        end
    end
end


"""
        grow_star(v::Integer,
                w::Integer,
                g::LightGraphs.AbstractGraph,
                firstNeighbor::Array{Tuple{Integer, Integer}, 1},
                set::DisjointSets{LightGraphs.Edge})

Subroutine to grow a 2-colored star after assigning a new color to the 
previously uncolored vertex v, by comparing it with the adjacent vertex w.
Disjoint set is used to store stars in sets, which are identified through key
edges present in g. 
"""
function grow_star(v::Integer,
                   w::Integer,
                   g::LightGraphs.AbstractGraph,
                   firstNeighbor::Array{Tuple{Integer, Integer}, 1},
                   set::DisjointSets{LightGraphs.Edge})
    edge = find_edge(g, v, w)
    push!(set, edge)
    p, q = firstNeighbor[color[w]]
    if p != v
        firstNeighbor[color[w]] = (v, w)
    else
        edge1 = find_edge(g, v, w)
        edge2 = find_edge(g, p, q)
        e1 = find_root(set, edge1)
        e2 = find_root(set, edge2)
        union!(set, e1, e2)
    end
end


"""
        merge_trees(v::Integer,
                w::Integer, 
                x::Integer,
                g::LightGraphs.AbstractGraph,
                set::DisjointSets{LightGraphs.Edge})

Subroutine to merge trees present in the disjoint set which have a
common edge.
"""
function merge_trees(v::Integer,
                    w::Integer, 
                    x::Integer,
                    g::LightGraphs.AbstractGraph,
                    set::DisjointSets{LightGraphs.Edge})
    edge1 = find_edge(g, v, w)
    edge2 = find_edge(g, w, x)
    e1 = find_root(set, edge1)
    e2 = find_root(set, edge2)
    if (e1 != e2)
        union!(set, e1, e2)
    end
end


"""
        find_edge(g::LightGraphs.AbstractGraph, v::Integer, w::Integer)

Returns an edge object of the type LightGraphs.Edge which represents the 
edge connecting vertices v and w of the undirected graph g
"""
function find_edge(g::LightGraphs.AbstractGraph, v::Integer, w::Integer)
    for v1 in outneighbors(g, v)
        if v1 == w
            e = LightGraphs.SimpleGraphs.SimpleEdge(v, w)
            return e
        end
    end
    throw(error("$v and $w are not connected in graph g"))
end

"""
        edge_index(g::LightGraphs.AbstractGraph, e::LightGraphs.Edge)

Returns an Integer value which uniquely identifies the edge e in graph
g. Used as an index in main function to avoid custom arrays with non-
numerical indices.
"""
function edge_index(g::LightGraphs.AbstractGraph, e::LightGraphs.Edge)
    edge_list = collect(edges(g))
    for i in 1:length(edge_list)
        if edge_list[i] == e
            return i
        end
    end
    return -1
end
