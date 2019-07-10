#BSC 0-based
using VertexSafeGraphs
using OffsetArrays

function color_graph(g::VSafeGraph)
    #number of vertices
    v = nv(g)

    #ordering of vertices of g in non increasing order of degree
    A_ = sort_by_degree(g)
    A = OffsetArray(A_, 0:length(A_)-1)

    #starting index
    start = 0

    #optimal color number
    opt = v + 1

    #current vertex to be colored
    x = A[0]


    #colors[-1] = 0
    colors_ = zeros(Int64, v)
    colors = OffsetArray(colors_, 0:length(colors_)-1)

    #U is set of free colors for current vertex
    U_ = zeros(Int64, 0)
    U = OffsetArray(U_, 0:-1)
    push!(U,1)

    #freeColors[x] is set of free colors for vertex x
    freeColors_ = [Vector{Int64}() for _ in 1:v]
    freeColors = OffsetArray(freeColors_, 0:v-1)
    freeColors[x] = copy(U)

    while(start >= 0)
        back = false
        for i = start:v
            if i > start
                x = uncolored_vertex_of_max_degree(A, F)

                #set of freeColors for x less than opt
                U = set_of_free_colors(x,A,colors,g, F, opt)

                #sort U non decreasing order
                sort(U)

            end

            if length(U) > 0
                k = U[0]
                F[x] = k

                #remove k from U
                deleteat!(U, 0)
                freeColors[x] = copy(U)

                if i-1==-1
                    l = 0
                else
                    l = colors[i-1]
                end

                colors[i] = max(k,l)
            else
                start = i-1
                back = true
                break
            end
        end

        if back
            if start >= 0
                x = A[start]
                F[x] = 0
                U = freeColors[x]
            end
        else
            Fopt = copy(F)
            opt = colors[v-1]

            #least index with F[A[i]] == opt
            i = least_index(F,A,opt)
            start = i-1
            if start < 0
                break
            end

            #uncolor all vertices A[i] where i >= start
            uncolor_all!(F,A,start)
            for i = 0:start
                x = A[i]
                U = freeColors[x]

                #remove from U all colors > opt
                U  = remove_from_U(U, opt)
                freeColors[x] = copy(U)
            end
        end
    end

    Fopt
end

function degree(G::VSafeGraph,z::Int64)
    return length(inneighbors(G,z))
end

function sort_by_degree(g::VSafeGraph)
    g1 = copy(g)
    g2 = copy(g)
    v = nv(g)
    sorted = zeros(Int64, 0)
    max_degree = -1
    max_degree_vertex = -1
    while (nv(g2) > 0)
        max_degree = -1
        max_degree_vertex = -1
        for vertex_i in vertices(g2)
            if degree(g1, vertex_i) > max_degree
                max_degree = degree(g1, vertex_i)
                max_degree_vertex = vertex_i
            end
        end
        push!(sorted, max_degree_vertex)
        rem_vertex!(g2, max_degree_vertex)
    end
    return sorted
end

function uncolored_vertex_of_max_degree(A, F)
    for i in A
        if F[i] == 0
            return i
        end
    end
end



function set_of_free_colors(x,A,colors,g,F,opt)
    index = -1

    freecolors_ = zeros(Int64, 0)
    freecolors = OffsetArray(freecolors_, 0:-1)

    for i in eachindex(A)
        if A[i] == x
            index = i
            break
        end
    end
    if index == 0
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

#least index i such that F[A[i]] == opt
function least_index(F,A,opt)
    for i in eachindex(A)
        if F[A[i]] == opt
            return i
        end
    end
end

 #uncolor all vertices A[i] where i >= start
function uncolor_all!(F,A,start)
    for i = start:length(A)-1
        F[A[i]] = 0
    end
end

#remove from U all colors >= opt
function remove_from_U!(U, opt)
    u_ = zeros(In64, 0)
    u = OffsetArray(u_, 0:-1)
    for i in U
        if i < opt
            push!(u, i)
        end
    end
    u
end

        
