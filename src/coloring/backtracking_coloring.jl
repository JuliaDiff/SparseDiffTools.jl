using LightGraphs

"""
    color_graph(g::LightGraphs, ::BacktrackingColor)

Returns a tight, distance-1 coloring of graph g
using the minimum number of colors possible (the
chromatic number of g, χ(g))
"""
function color_graph(g::LightGraphs.AbstractGraph, ::BacktrackingColor)
    println("function called")
    v = nv(g)

    A = sort_by_degree(g)

    #F is the coloring of vertices, 0 means uncolored
    #Fopt is the optimal coloring of the graph
    F = zeros(Int32, v)
    Fopt= zeros(Int32, v)

    #start index
    start = 1
    println("start = $start")

    #optimal color number
    opt = v + 1

    #current vertex to be colored
    x = A[1]

    #colors[j] = number of colors in A[0]...A[j]
    #assume colors[0] = 1
    colors = zeros(Int32, v)

    #set of free colors
    U = zeros(Int32, 0)
    push!(U, 1)
    print("1. U = ")
    println(U)

    #set of free colors of x
    freeColors = [Vector{Int64}() for _ in 1:v]
    freeColors[x] = copy(U)

    while (start >= 1)
        #println("Running while loop, start = $start")
        back = false
        for i = start:v
            #println("running for loop, i = $i")
            if i > start
                #println("i > start, entering if block (1)")
                x = uncolored_vertex_of_maximal_degree(A,F)
                #println("x = $x")
                U = free_colors(x, A, colors, F, g, opt)
                #print("U = ")
                #println(U)
                sort!(U)
                #println("sorted U = ")
                #println(U)
            end
            if length(U) > 0
                #println("length of U > 0, entering if block (2)")
                k = U[1]
                #println("k = $k")
                F[x] = k
                #println("set F[$x] = $k")
                cp = F[x]
                #println("F[$x] = $cp")
                deleteat!(U, 1)
                #print("U = ")
                #println(U)
                freeColors[x] = copy(U)
                #print("freeColors[$x] = ")
                #println(U)
                if i==1
                    l = 0
                    #println("l = 0")
                else
                    l = colors[i-1]
                    #println("l = $l")
                end
                colors[i] = max(k, l)
                #ck = colors[i]
                #println("colors[$i] = $ck")
            else
                #println("else condition, block (3)")
                start = i-1
                #println("start = $start")
                back = true
                #println("back = true")
                break
            end
        end
        if back
            #println("back is true, if block (4)")
            if start >= 1
                #println("start = $start >= 1, if block (5)")
                x = A[start]
                #println("x = $x")
                F[x] = 0
                #println("F[$x] = 0")
                U = freeColors[x]
                #print("U = ")
                #println(U)
            end
        else
            #println("else condition, back is not true block (6)")
            Fopt = copy(F)
            #print("Fopt = ")
            #println(Fopt)
            opt = colors[v-1]
            #println("opt = $opt")
            i = least_index(F,A,opt)
            #println("i = $i")
            start = i-1
            #println("start = $start")
            if start < 1
                #println("start < 1, so breaking")
                break
            end

            #uncolor all vertices A[i] with i >= start
            uncolor_all!(F, A, start)
            #println("uncoloring")
            #try start+1 instead
            for i = 1:start+1
                x = A[i]
                U = freeColors[x]
                #remove colors >= opt from U
                #print("U = ")
                #println(U)
                #println("removing from U all colors >= $opt")
                U = remove_higher_colors(U, opt)
                #print("U = ")
                #println(U)
                freeColors[x] = copy(U)
            end
        end

    end

    Fopt
end

"""
    sort_by_degree()

sort and store the vertices of graph g in
non-increasing order of their degrees
"""
function sort_by_degree(g::LightGraphs.AbstractGraph)
    vs = vertices(g)
    degrees = (LightGraphs.degree(g, v) for v in vs)
    vertex_pairs = collect(zip(vs, degrees))
    sort!(vertex_pairs, by = p -> p[2], rev = true)
    [v[1] for v in vertex_pairs]
end

"""
    uncolored_vertex_of_maximal_degree(A,F)

Returns an uncolored vertex from the graph which has maximum
degree
"""
function uncolored_vertex_of_maximal_degree(A,F)
    for v in A
        if F[v] == 0
            return v
        end
    end
end


"""
    free_colors()

returns set of free colors of x which are less
than optimal color number (opt)
"""
function free_colors(x, A, colors, F, g, opt)
    index = -1

    freecolors = zeros(Int64, 0)

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

    freecolors

end

"""
    least_index()

returns least index i such that color of vertex
A[i] == opt (optimal color number)
"""
function least_index(F,A,opt)
    for i in eachindex(A)
        if F[A[i]] == opt
            return i
        end
    end
end

"""
    uncolor_all()

uncolors all vertices A[i] where
i >= start
"""
function uncolor_all!(F, A, start)
    for i = start:length(A)
        F[A[i]] = 0
    end
end

"""
    remove_higher_colors()

remove all the colors >= opt (optimal color number)
from the set of colors U
"""
function remove_higher_colors(U, opt)
    if length(U) == 0
        return U
    end
    u = zeros(Int32, 0)
    for color in U
        if color < opt
            push!(u, color)
        end
    end
    u
end
