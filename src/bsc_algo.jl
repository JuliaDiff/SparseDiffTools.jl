#Backtracking Sequential Coloring algorithm
include ("custom_graph.jl")

function bsc_color(G::CGraph)
    V = length(vertices(G))
    A = sorted_vertices(G)
    U = zeros(Int64, 0)
    F = zeros(Int64, V)
    Fopt = zeros(Int64, V)
    freeColors = [Vector{Int64}() for _ in 1:V]
    colors = zeros(Int64, V)
    start = 1
    optColorNumber = V + 1
    x = A[1]
    push!(U,1)
    freeColors[x] = copy(U)

    while start >= 1
        back = false
        for i = start:V
            if i > start
                x = uncolormaxdegree(F,A, G)
                U = free_colors(x, F, G, optColorNumber)
                sort(U)
            end
            if length(U) > 0
                k = U[1]
                F[x] = k
                deleteat!(U, 1)
                freeColors[x] = copy(U)
                if i-1 == 0
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
                F[x] = 0
                U = freeColors[x]
            end
        else
            Fopt = F
            optColorNumber = colors[V-1]
            i = least_index(A, F, optColorNumber, G)
            start = i-1
            if start <= 0
                break
            end
            uncolor_all(F,A,start,G)
            for i = 1:start
                x = A[i]
                U = freeColors[x]
                U = remove_colors(U, optColorNumber)
                freeColors[x] = copy(U)
            end
        end
        return Fopt
    end

end

function degree(G::CGraph,z::Int64)
    return length(neighbors(G,z))
end

function sorted_vertices(G::CGraph)
    V = length(vertices(G))
    marked = zeros(Int64,V)
    sv = zeros(Int64,0)
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

function least_index(A::Array{Int64, 1}, F::Array{Int64,1}, optColorNumber::Int64, G::CGraph)
    for i = 1:length(G.vertices)
        if F[A[i]] == optColorNumber
            return i
        end
    end
end

#uncolor all vertices A[i] with i >= start
function uncolor_all(F::Array{Int64,1}, A::Array{Int64,1}, start::Int64, G::CGraph)
    for i = start:length(G.vertices)
        F[A[i]] = 0
    end
end

function remove_colors(U::Array{Int64,1}, optColorNumber::Int64)
    modified_U = zeros(Int64,0)
    for i = 1:length(U)
        if U[i] < optColorNumber
            push!(mmodified_U, U[i])
        end
    end
    return modified_U
end

function uncolormaxdegree(F::Array{Int64, 1},A::Array{Int64,1}, G::CGraph)
    colors = zeros(Int64,0)
    max_colors = -1
    max_color_index = -1
    for i = 1:length(vertices(G))
        if F[i] != 0
            for j in neighbors(G,i)
                if F[j] != 0 && F[j] in colors == false
                    push!(colors, F[j])
                end
            end
            if length(colors) > max_colors
                max_colors = length(colors)
                max_color_index = i
            end
        end
        colors = zeros(Int64,0)
    end
    for i = 1:length(vertices(G))
        if A[i] == max_color_index
            return i
        end
    end

end
