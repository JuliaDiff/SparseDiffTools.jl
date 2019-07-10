using OffsetArrays
using VertexSafeGraphs
"""
    BSCColor

    Backtracking Sequential Coloring algorithm
"""
function color_graph(G::VSafeGraph)
    V = nv(G)
    F = zeros(Int64, V)
    F_opt = zeros(Int64, V)
    freeColors = [Vector{Int64}() for _ in 1:V] #set of free colors for each vertex
    colors = zeros(Int64, V)
    U = zeros(Int64, 0) #stores set of free colors


    A = sorted_vertices(G)
    #A = OffsetArray(A0, 0:length(A0)-1)

    start = 1
    optColorNumber = V + 1
    x = A[1]
    #colors[-1] = 0
    push!(U, 1)
    freeColors[x] = copy(U)

    while (start >= 0)
        back = false
        for i = 1:V
            if i > start
                x = find_uncolored_vertex(A, F)
                U = free_colors(x,optColorNumber, colors, A, F, g)
                sort(U)
            end
            if length(U) > 0
                k = U[1]
                F[x] = k
                deleteat!(U,1)
                freeColors[x] = copy(U)
                if i-1==0
                    l = 0
                else
                    l = colors[i-1]
                end
                colors[i] = max(k,l)
            else
                start = i - 1
                back = true
                break
            end
        end
        if back
            if start >= 1
                x = A[start]
                F[x] = 0 #uncolor x
                U = freeColors[x]
            end
        else
            F_opt = F
            optColorNumber = colors[V-1]
            i = least_index(A,F,optColorNumber,G)
            start = i - 1
            if start < 1
                break #leave the while loop
            end
            uncolor_all(F, A, start, G)
            for i = 1:start
                x  = A[i]
                U = freeColors[x]
                U = remove_colors(U, optColorNumber)
                freeColors[x] = copy(U)
            end
        end
    end
    return F_opt
end


"""
    vertex_degree(G,z)

Find the degree of the vertex z which belongs to the graph G.
"""
function degree(G::VSafeGraph,z::Int64)
    return length(inneighbors(G,z))
end


function sorted_vertices(G::VSafeGraph)
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

#find uncolored vertex of maximal degree of saturation
function find_uncolored_vertex(sv::Array{Int64,1}, F::Array{Int64,1})
    for i in sv
        if F[i] == 0
            return i
        end
    end
end

#set of free colors of x, which are < optColorNumber
function free_colors(x, ocn, colors, A, F, g)
    index = -1
    freecolors = zeros(Int64, 0)
    for i = 1:length(A)
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

#least index with F(A[i]) = optColorNumber
function least_index(A::Array{Int64, 1}, F::Array{Int64,1}, optColorNumber::Int64, G::VSafeGraph)
    for i = 1:nv(G)
        if F[A[i]] == optColorNumber
            return i
        end
    end
end

#uncolor all vertices A[i] with i >= start
function uncolor_all(F::Array{Int64,1}, A::Array{Int64,1}, start::Int64, G::VSafeGraph)
    for i = start:nv(G)
        F[A[i]] = 0
    end
end

#remove from U all colors >= optColorNumber
function remove_colors(U::Array{Int64,1}, optColorNumber::Int64)
    modified_U = zeros(Int64,0)
    for i = 1:length(U)
        if U[i] < optColorNumber
            push!(modified_U, U[i])
        end
    end
    return modified_U
end
