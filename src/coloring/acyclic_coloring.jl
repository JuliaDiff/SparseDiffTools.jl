using LightGraphs

"""
        color_graph(g::LightGraphs.AbstractGraphs, :: AcyclicColoring)

Returns a coloring vector following the acyclic coloring rules (1) the coloring 
corresponds to a distance-1 coloring, and (2) vertices in every cycle of the 
graph are assigned at least three distinct colors. This variant of coloring is 
called acyclic since every subgraph induced by vertices assigned any two colors
is a collection of trees—and hence is acyclic.
"""
function color_graph(g::LightGraphs.AbstractGraphs, ::AcyclicColoring)
    
    color = zeros(Int, nv(g))
    forbiddenColors = zeros(Int, nv(g))

    for v in vertices(g)
        #enforces the first condition of acyclic coloring
        for w in outneighbors(g, v)
            forbiddenColors[color[w]] = v
        #enforces the second condition of acyclic coloring
        for w in outneighbors(g, v)
            if color[w] != 0 #colored neighbor
                for x in outneighbors(w)
                    if color[x] != 0 #colored x
                        if forbiddenColors[color[x]] != v
                            prevent_cycle(v, w, x)
                        end
                    end
                end
            end
        end

        color[v] = min_index(forbiddenColors, v)

        #grow star for every edge connecting colored vertices v and w
        for w in outneighbors(g, v)
            if color[w] != 0
                growStar(v, w)
            end
        end

        #merge the newly formed stars into existing trees if possible
        for w in outneighbors(g, v)
            if color[w] != 0
                for x in outneighbors(g, w)
                    if color[x] != 0 && x != v
                        if color[x] == color[v]
                            mergeTrees(v, w, x)
                        end
                    end
                end
            end
        end 
    end
end

"""
        prevent_cycle()

"""
function prevent_cycle(v::Integer,
                       w::Integer, 
                       x::Integer, 
                       forbiddenColors::AbstractVector{<:Integer},
                       firstVisitToTree)
    e = Find(w, x)
    p, q = firstVisitToTree[e]
    if p != v
        firstVisitToTree[e] = (v, w)
    else if q != w
        forbiddenColors[color[x]] = v
    end
end

"""
        min_index(forbiddenColors::AbstractVector{<:Integer}, v::Integer)

Returns min{i > 0 such that forbiddenColors[i] != v}
"""            
function min_index(forbiddenColors, v)
    for i = 1:length(forbiddenColors)
        if forbiddenColors[i] != v
            return i
        end
    end
end

function growStar(v:: Integer, w::Integer)
end

function mergeTrees(v:: Integer, w::Integer, x::Integer)
end