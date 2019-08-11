using LightGraphs

"""
    color_graph(g::LightGraphs, ::BacktrackingColor)

Returns a tight, distance-1 coloring of graph g
using the minimum number of colors possible (i.e.
the chromatic number of graph, χ(g))
"""
function color_graph(g::LightGraphs.AbstractGraph, ::BacktrackingColor)
    v = nv(g)

    #A is list of vertices in non-increasing order of degree
    A = sort_by_degree(g)

    #F is the coloring of vertices, 0 means uncolored
    #Fopt is the optimal coloring of the graph
    F = zeros(Int, v)
    Fopt= zeros(Int, v)

    start = 1

    #optimal chromatic number
    opt = v + 1

    #current vertex to be colored
    x = A[1]

    #colors[j] = number of colors in A[0]...A[j]
    #assume colors[0] = 1
    colors = zeros(Int, v)

    #set of free colors
    U = zeros(Int, 0)
    push!(U, 1)

    #set of free colors of x
    freeColors = [Vector{Int}() for _ in 1:v]
    freeColors[x] = copy(U)

    while (start >= 1)

        back = false
        for i = start:v
            if i > start
                x = uncolored_vertex_of_maximal_degree(A,F)
                U = free_colors(x, A, colors, F, g, opt)
                sort!(U)
            end
            if length(U) > 0
                k = U[1]
                F[x] = k
                cp = F[x]
                deleteat!(U, 1)
                freeColors[x] = copy(U)
                if i==1
                    l = 0
                else
                    l = colors[i-1]
                end
                colors[i] = max(k, l)
            else
                start = i-1
                back = true
                break
            end
        end

        if back
            if start >= 1
                x = A[start]
                F[x] = 0
                U = freeColors[x]
            end
        else
            Fopt = copy(F)
            opt = colors[v-1]
            i = least_index(F,A,opt)
            start = i-1
            if start < 1
                break
            end

            #uncolor all vertices A[i] with i >= start
            uncolor_all!(F, A, start)

            for i = 1:start+1
                x = A[i]
                U = freeColors[x]

                #remove colors >= opt from U
                U = remove_higher_colors(U, opt)
                freeColors[x] = copy(U)
            end
        end
    end
    return Fopt
end

"""
    sort_by_degree(g::LightGraphs.AbstractGraph)

Returns a list of the vertices of graph g sorted
in non-increasing order of their degrees
"""
function sort_by_degree(g::LightGraphs.AbstractGraph)
    vs = vertices(g)
    degrees = (LightGraphs.degree(g, v) for v in vs)
    vertex_pairs = collect(zip(vs, degrees))
    sort!(vertex_pairs, by = p -> p[2], rev = true)
    return [v[1] for v in vertex_pairs]
end

"""
    uncolored_vertex_of_maximal_degree(A::AbstractVector{<:Integer},F::AbstractVector{<:Integer})

Returns an uncolored vertex from the partially
colored graph which has the highest degree
"""
function uncolored_vertex_of_maximal_degree(A::AbstractVector{<:Integer},F::AbstractVector{<:Integer})
    for v in A
        if F[v] == 0
            return v
        end
    end
end


"""
    free_colors(x::Integer,
                A::AbstractVector{<:Integer},
                colors::AbstractVector{<:Integer},
                F::Array{Integer,1},
                g::LightGraphs.AbstractGraph,
                opt::Integer)

Returns set of free colors of x which are less
than optimal chromatic number (opt)

Arguments:

x: Vertex who's set of free colors is to be calculated
A: List of vertices of graph g sorted in non-increasing order of degree
colors: colors[i] stores the number of distinct colors used in the
        coloring of vertices A[0], A[1]... A[i-1]
F: F[i] stores the color of vertex i
g: Graph to be colored
opt: Current optimal number of colors to be used in the coloring of graph g
"""
function free_colors(x::Integer,
                    A::AbstractVector{<:Integer},
                    colors::AbstractVector{<:Integer},
                    F::Array{Integer,1},
                    g::LightGraphs.AbstractGraph,
                    opt::Integer)
    index = -1

    freecolors = zeros(Int, 0)

    for i in eachindex(A)
        if A[i] == x
            index = i
            break
        end
    end

    if index == 1
        colors_used = 0
    else
        colors_used = colors[index-1]
    end

    colors_used += 1
    for c = 1:colors_used
        c_allowed = true
        for w in inneighbors(g, x)
            if F[w] == c
                c_allowed = false
                break
            end
        end
        if c_allowed && c < opt
            push!(freecolors, c)
        end
    end

    return freecolors

end

"""
    least_index(F::AbstractVector{<:Integer}, A::AbstractVector{<:Integer}, opt::Integer)

Returns least index i such that color of vertex
A[i] is equal to `opt` (optimal chromatic number)
"""
function least_index(F::AbstractVector{<:Integer}, A::AbstractVector{<:Integer}, opt::Integer)
    for i in eachindex(A)
        if F[A[i]] == opt
            return i
        end
    end
end

"""
    uncolor_all(F::AbstractVector{<:Integer}, A::AbstractVector{<:Integer}, start::Integer)

Uncolors all vertices A[i] where i is
greater than or equal to start
"""
function uncolor_all!(F::AbstractVector{<:Integer}, A::AbstractVector{<:Integer}, start::Integer)
    for i = start:length(A)
        F[A[i]] = 0
    end
end

"""
    remove_higher_colors(U::AbstractVector{<:Integer}, opt::Integer)

Remove all the colors which are greater than or
equal to the `opt` (optimal chromatic number) from
the set of colors U
"""
function remove_higher_colors(U::AbstractVector{<:Integer}, opt::Integer)
    if length(U) == 0
        return U
    end
    u = zeros(Int, 0)
    for color in U
        if color < opt
            push!(u, color)
        end
    end
    return u
end
