"""
    BSCColor

    Backtracking Sequential Coloring algorithm
"""
function color_graph(G::VSafeGraph,::BSCColor)
    V = nv(G)
    F = zeros(Int, V)
    freeColors = [Vector{Int}() for _ in 1:V] #set of free colors for each vertex
    U = zeros(Int, 0) #stores set of free colors
    #F = zeros(Int, 0) #stores final coloring of vertices

    sorted_vertices = order_by_degree(G)
    start = 1
    optColorNumber = V + 1
    x = sorted_vertices[1]
    colors[0] = 0
    push!(U, 1)
    freeColors[x] = U

    while (start >= 1)
        back = false
        for i = 1:V
            if i > start
                x = find_uncolored_vertex(sorted_vertices, F)
                U = free_colors(x,F,G,optColorNumber)
                sort(U)
            end
            if length(U) > 0
                k = U[1]
                F[x] = k
                deleteat!(U,1)
                freeColors[x] = copy(U)
                l = colors[i-1]
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
            i = least_index(sorted_vertices,optColorNumber,G)
            start = i - 1
            if start < 1
                break #leave the while loop
            end
            uncolor_all(F, sorted_vertices, start, G)
            for i = 0:start
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
function degree(G::VSafeGraph,z::Int)
    return length(inneighbors(G,z))
end


function sorted_vertices(G::VSafeGraph)
    V = nv(G)
    marked = zeros(Int,V)
    sv = zeros(Int,0)
    max_degree = -1
    max_degree_vertex = -1
    for i = 1:V
        max_degree = -1
        max_degree_vertex = -1
        for j = 1:V
            if j != i
                if degree(G,j) > max_degree && marked[j] == 0
                    max_degree = degree(G,j)
                    max_degree_vertex = j
                end
            end
        end
        push!(sv,max_degree_vertex)
        marked[max_degree_vertex] = 1
    end
    return sv
end

#find uncolored vertex of maximal degree of saturation
function find_uncolored_vertex(sv::Array{Int,1}, G::VSafeGraph)
    colors = zeros(Int,0)
    max_colors = -1
    max_color_index = -1
    for i = 1:nv(G)
        if F[i] != 0
            for j in inneighbors(G,i)
                if F[j] != 0 && F[j] in colors == false
                    push!(colors, F[j])
                end
            end
            if length(colors) > max_colors
                max_colors = length(colors)
                max_color_index = i
            end
        end
        colors = zeros(Int,0)
    end
    for i = 1:nv(G)
        if A[i] == max_color_index
            return i
        end
    end

end

#set of free colors of x, which are < optColorNumber
function free_colors(x::Int, F::Array{Int,1}, G::VSafeGraph, max_color::Int)
    colors = zeros(Int,0)
    for color in 1:max_color
        present = true
        for y in inneighbors(G,x)
            if F[y] == color
                present = false
                break
            end
        end
        if present
            push!(colors,color)
        end
    end
    return colors
end

#least index with F(A[i]) = optColorNumber
function least_index(A::Array{Int, 1}, F::Array{Int,1}, optColorNumber::Int, G::VSafeGraph)
    for i = 1:nv(G)
        if F[A[i]] == optColorNumber
            return i
        end
    end
end

#uncolor all vertices A[i] with i >= start
function uncolor_all(F::Array{Int,1}, A::Array{Int,1}, start::Int, G::VSafeGraph)
    for i = start:nv(G)
        F[A[i]] = 0
    end
end

#remove from U all colors >= optColorNumber
function remove_colors(U::Array{Int,1}, optColorNumber::Int)
    modified_U = zeros(Int,0)
    for i = 1:length(U)
        if U[i] < optColorNumber
            push!(mmodified_U, U[i])
        end
    end
    return modified_U
end
