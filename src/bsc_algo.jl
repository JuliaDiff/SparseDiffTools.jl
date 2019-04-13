#Backtracking Sequential Coloring algorithm
include ("custom_graph.jl")

function bsc_color(G::CGraph)
    V = num_vertices(G)
    colors = zeros(Int64, V)
    freeColors = [Vector{Int64}() for _ in 1:V] #set of free colors for each vertex
    U = zeros(Int64, 0) #stores set of free colors
    F = zeros(Int64, 0) #stores final coloring of vertices

    sorted_vertices = order_by_degree(G)
    start = 0
    optColorNumber = V + 1
    x = sorted_vertices[1]
    colors[]
    push!(U, 1)
    freeColors[x] = U

    while (start >= 0)
        back = false
        for i = 1:(V-1)
            if i > start
                x = find_uncolored_vertex(sorted_vertices, F)
                U = free_colors(x,F,optColorNumber)
                sort_inc!(U)
            end
            if length(U) > 0
                k = U[1]
                F[x] = k
                deleteat!(U,1)
                freeColors[x] = U
                l = colors[i-1]
                colors[i] = max(k,l)
            else
                start = i - 1
                back = true
                break
            end
        if back
            if start >= 0
                x = A[start]
                F[x] = 0 #uncolor x
                U = freeColors[x]
            end
        else
            F_opt = F
            optColorNumber = colors[V-1]
            i = least_index(A,optColorNumber)
            start = i - 1
            if start < 0
                break
            end
            uncolor_all(A, i, start)
            for i = 0:start
                x  = A[i]
                U = freeColors[x]
                remove_colors(U, optColorNumber)
                freeColors[x] = U
            end
        end
    end
    return F_opt
end
