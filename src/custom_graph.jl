#=
    CustomGraph Module for representation and storage of undirected graph

AUTHOR: Pankaj Mishra
DATE: 27 March 2019

Although powerful and robust graph packages already exist for Julia, I found out
that the two most common ones, namely Graphs.jl and LightGraphs.jl were not very
useful for the algorithm I wanted to implement.

While the ability to delete edges and vertices is completely absent in case of
Graphs.jl, it is present in LightGraphs.jl. However, I found that deleting a
vertex using the rem_edge! function in Light Graph module would automatically
rearrange the order of vertices without respecting the original sequence.

Therefore, I decided to write this simple yet complete module for the
representation of an undirected graph using an adjacency list which allows for
the addition of vertices that need not be ordered strictly from 1 to nv, or
force the rearrangement of vertices upon deletion of one.

While the function names are descriptive and clear to understand, a small
documentation is provided with each function to make its purpose very clear.
=#

"""
    CGraph

A mutable type that can be used to store a graph in the form of an
adjacency list. Has 2 input parameters:
vertices: a vector that stores the list of current vertices in the graph
adj_list: a vector of vectors that stores the neighbors of vertex i in the
          vector stored at adj_list[i]
"""
mutable struct CGraph
    vertices::Array{Int64,1}
    adj_list::Array{Array{Int64,1},1}
end

"""
    add_vertex!(G,x)

Add a vertex x to the graph G.
"""
function add_vertex!(G::CGraph, x::Int64)
    push!(G.vertices,x)
end

"""
    add_edge!(G,src,dst)

Add an edge between the vertices src and dst that belong to the
graph G.
"""
function add_edge!(G::CGraph, src::Int64, dst::Int64)
    push!(G.adj_list[src],dst)
    push!(G.adj_list[dst],src)
end


"""
    neighbors(G,x)

Get a list of all the vertices of graph G that are connected to
vertex x directly with an edge.
"""
function neighbors(G::CGraph, x::Int64)
    return G.adj_list[x]
end


"""
    vertices(G)

Get a list of all the vertices present in the graph G.
"""
function vertices(G::CGraph)
    return G.vertices
end


"""
    has_edge(G,src,dst)

Return true if the vertices src and dst are connected by an edge, false
otherwise.
"""
function has_edge(G::CGraph, src::Int64, dst::Int64)
    for i in G.adj_list[src]
        if i == dst
            return true
        end
    end
    return false
end


"""
    num_edges(G)

Return the total number of edges in the graph.
"""
function num_edges(G::CGraph)
    num = 0
    for i in vertices(G)
        num += length(G.adj_list[i])
    end
    num /= 2
    return num
end

"""
    has_vertex(G,x)

Return true if the graph G contains the vertex x, false otherwise.
"""
function has_vertex(G::CGraph, x::Int64)
    for i in G.vertices
        if i == x
            return true
        end
    end
    return false
end

"""
    rem_edge!(G,src,dst)

Remove the edge between the vertices src and dst in the graph G.
"""
function rem_edge!(G::CGraph, src::Int64, dst::Int64)
    src_index = indexin([src],G.adj_list[dst])[1]
    dst_index = indexin([dst],G.adj_list[src])[1]
    deleteat!(G.adj_list[src],dst_index)
    deleteat!(G.adj_list[dst],src_index)
end


"""
    rem_vertex!(G,x)

Remove the vertex x from graph G, and delete any edges that might
be connected to it.
"""
function rem_vertex!(G::CGraph, x::Int64)
    #go through all neighbors of x and remove all edges first
    nbs = length(neighbors(G,x))
    for i = 1:nbs
        nb = (G.adj_list[x])[1]
        rem_edge!(G,nb,x)
    end
    x_index = indexin([x],G.vertices)[1]
    deleteat!(G.vertices,x_index)
end
