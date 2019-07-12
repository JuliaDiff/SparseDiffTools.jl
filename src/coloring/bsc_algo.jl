#bsc 1-based indexing
using LightGraphs

function color_graph(g::LightGraphs.AbstractGraph)

    #number of vertices in g
    v = nv(g)

    #A is order of vertices in non-increasing order of degree
    A = sort_by_degree(g)

    #F is the coloring of vertices, 0 means uncolored
    #Fopt is the optimal coloring of the graph
    F = zeros(Int32, v)
    Fopt= zeros(Int32, v)

    #start index
    start = 1

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

    #set of free colors of x
    freeColors = [Vector{Int64}() for _ in 1:v]
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
                deleteat!(U, 1)
                freeColors[x] = copy(U)
                if i==1
                    l = 0
                else
                    l = colors[i-1]
                colors[i] = max(k, l)
                end
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

            #try start+1 instead
            for i = 1:start+1
                x = A[i]
                U = freeColors[x]
                #remove colors >= opt from U
                U = remove_higher_colors(U, opt)
                freeColors[x] = copy(U)
            end
        end

    end

    F
end


function sort_by_degree(g::LightGraphs.AbstractGraph)
    vs = vertices(g)
    degrees = (LightGraphs.degree(g, v) for v in vs)
    vertex_pairs = collect(zip(vs, degrees))
    sort!(vertex_pairs, by = p -> p[2], rev = true)
    [v[1] for v in vertex_pairs]
end


function uncolored_vertex_of_maximal_degree(A,F)
    for v in A
        if F[v] == 0
            return v
        end
    end
end


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
        if c_allowed && c < ocn
            push!(freecolors, c)
        end
    end

    freecolors

end

function least_index(F,A,opt)
    for i in eachindex(A)
        if F[A[i]] == opt
            return i
        end
    end
end

function uncolor_all!(F, A, start)
    for i = start:length(A)
        F[A[i]] = 0
    end
end


function remove_higher_colors(U, opt)
    u = zeros(Int32, 0)
    for color in U
        if color < opt
            push!(u, color)
        end
    end
end
