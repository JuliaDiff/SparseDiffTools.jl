using LightGraphs

"""
        detect_cycles(g::LightGraphs.AbstractGraph)

Prints all the cycles present in undirected graph g
"""
function detect_cycles(g::LightGraphs.AbstractGraph)
    color = zeros(Int64, 100)
    par = zeros(Int64, 100)

    mark = zeros(Int64, 100)

    cycleNumber = 0
    edges = ne(g)

    cycleNumber = dfs_cycle(1, 1, color, mark, par, cycleNumber)
    print_cycles(edges, mark, cycleNumber)

end

function dfs_cycle(u::Integer,
                    p::Integer,
                    color::AbstractArray{<:Integer, 1},
                    mark::AbstractVector{<:Integer},
                    par::AbstractVector{<:Integer},
                    cycleNumber::Integer)
    if color[u] == 2
        return cycleNumber
    end

    if color[u] == 1
        cycleNumber += 1
        cur = p
        mark[cur] = cycleNumber

        while (cur != u)
            cur = par[cur]
            mark[cur] = cycleNumber
        end
        return cycleNumber
    end

    par[u] = p
    color[u] = 1

    for v in outneighbors(g, u)
        if v == par[u]
            continue
        end
        cycleNumber = dfs_cycle(v, u, color, mark, par, cycleNumber)
    end

    color[u] = 2
end

function print_cycles(edges::Integer,
                    mark::AbstractVector{<:Integer},
                    cycleNumber::Integer)

    cycles = Array{Array{Integer,1}}(undef, 5)

    for i in 1:edges
        if mark[i] != 0
            push!(cycles[mark[i]], i)
        end
    end

    for i in 1:cycleNumber
        println("Cycle number $i")
        for i in cycles[i]
            print("$i ")
        end
    end

end
