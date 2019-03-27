include("custom_graph.jl")

"""
    colorGraph(g)

Find a coloring of the graph g such that no two vertices connected
by an edge have the same color.
"""
function colorGraph(G::CGraph)
    colornumber = 0
    V = num_vertices(G)
    colors = zeros(Int64,V)

    while (V > 0)
        x = max_degree_vertex(G)
        colornumber = colornumber + 1
        colors[x] = colornumber
        nn = non_neighbors(G,x)
        while (length(nn) > 0)
            maxcn = -1
            ydegree = -1
            for z in nn
                cn = length_common_neighbor(G,z,x)
                if (cn > maxcn) || (cn == maxcn && vertex_degree(G,z) < ydegree)
                    y = z
                    ydegree = vertex_degree(G,y)
                    maxcn = cn
                end
            end
            if (maxcn == 0)
                y = max_degree_vertex(G,nn)
            end
            colors[y] = colornumber
            contract!(G,y,x)
            nn = non_neighbors(G,x)
        end
        rem_vertex!(G,x)
        V = num_vertices(G)
    end
    return colors
end

"""
    num_vertices(G)

Find the total number of vertices present in graph G.
"""
function num_vertices(G::CGraph)
    return length(vertices(G))
end

"""
    max_degree_vertex(G, nn)

Find the vertex in the group nn of vertices belonging to the
graph G which has the highest degree.
"""
function max_degree_vertex(G::CGraph,nn::Array{Int64,1})
    max_degree = -1
    max_degree_vertex = -1
    for v in nn
        v_degree = length(neighbors(G,v))
        if v_degree > max_degree
            max_degree = v_degree
            max_degree_vertex = v
        end
    end
    return max_degree_vertex
end

"""
    max_degree_vertex(G)

Find the vertex in graph with highest degree.
"""
function max_degree_vertex(G::CGraph)
    max_degree = -1
    max_degree_vertex = -1
    for v in vertices(G)
        v_degree = length(neighbors(G,v))
        if v_degree > max_degree
            max_degree = v_degree
            max_degree_vertex = v
        end
    end
    return max_degree_vertex
end

"""
    non_neighbors(G,x)

Find the set of vertices belonging to the graph G which do
not share an edge with the vertex x.
"""
function non_neighbors(G::CGraph, x::Int64)
    num_non_neighbors = num_vertices(G) - vertex_degree(G,x) - 1
    nn = zeros(Int64,num_non_neighbors)
    index = 1

    for i in vertices(G)
        if ((indexin([i], neighbors(G,x)))[1] == nothing) && (i != x)
            nn[index] = i
            index += 1
        end
    end
    return nn
end

"""
    length_common_neighbor(G,z,x)

Find the number of vertices that share an edge with both the
vertices z and x belonging to the graph G.
"""
function length_common_neighbor(G::CGraph,z::Int64, x::Int64)
    z_neighbors = neighbors(G,z)
    x_neighbors = neighbors(G,x)
    common_vertices = indexin(z_neighbors, x_neighbors)
    num_common_vertices = 0
    for i in common_vertices
        if i != nothing
            num_common_vertices += 1
        end
    end
    return num_common_vertices
end

"""
    vertex_degree(G,z)

Find the degree of the vertex z which belongs to the graph G.
"""
function vertex_degree(G::CGraph,z::Int64)
    return length(neighbors(G,z))
end


"""
    contract!(G,y,x)

Contract the vertex y to x, both of which belong to graph G, that is
delete vertex y and join x with the neighbors of y if they are not
already connected with an edge.
"""
function contract!(G::CGraph,y::Int64, x::Int64)
    for v in neighbors(G,y)
        if has_edge(G,v,x) == false
            add_edge!(G,v,x)
        end
    end
    rem_vertex!(G,y)

end
