function color_graph(g::LightGraphs.AbstractGraph, ::AcyclicColoring)
    color = zeros(Int, nv(g))
    set = DisjointSets{Int}([])

    first_visit_to_tree = Array{Tuple{Int, Int}, 1}(undef, ne(g))
    first_neighbor = Array{Tuple{Int, Int}, 1}(undef, ne(g))

    forbidden_colors = zeros(Int, nv(g))

    for v in vertices(g)
        println(">>>\nOUTER LOOP")
        println(">>> v = $v")
        println(">>> first block")
        for w in outneighbors(g, v)
            println(">>>     w = $w")
            if color[w]!=0
                wc = color[w]
                println(">>>     $w has nonzero color = $wc")
                println(">>>     setting forbidden color[$wc] = $v")
                forbidden_colors[color[w]] = v
            end
        end

        println(">>> second block")
        for w in outneighbors(g, v)
            println(">>>     w = $w")
            if color[w]!=0
                wc = color[w]
                println(">>>     $w has nonzero color = $wc")
                for x in outneighbors(g, w)
                    println(">>>          x = $x")
                    wx = color[x]
                    println(">>>          $x has color = $wx")
                    if color[x]!=0
                        println(">>>          $wx != 0")
                        fbc = forbidden_color[color[x]]
                        println(">>>          forbidden color[$wx] = $fbc")
                        if forbidden_colors[color[x]] != v
                            println(">>>          $fbc != $v")
                            println(">>>          calling prevent cycle with $v, $w, $x")
                            prevent_cycle!(v, w, x, g, set, first_visit_to_tree, forbidden_colors,color)
                        end
                    end
                end
            end
        end

        println(">>> third block")
        color[v] = min_index(forbidden_colors, v)
        vc = color[v]
        println(">>> color of v = $vc")
        for w in outneighbors(g, v)
            println(">>>     w = $w")
            if color[w]!=0
                println(">>>     calling grow star for v = $v, w = $w")
                grow_star!(v, w, g, set,first_neighbor,color)
            end
        end

        println(">>> fourth block")
        for w in outneighbors(g, v)
            println(">>>     w = $w"
            if color[w]!=0
                wc = color[w]
                println(">>>     $w has non zero color = $wc")
                for x in outneighbors(g, w)
                    wx = color[x]
                    println(">>>          x = $x")
                    if color[x]!=0 && x!=v
                        println(">>>          $x has nonzero color = $wx")
                        if color[x]==color[v]
                            merge_trees!(v,w,x,g,set)
                        end
                    end
                end
            end
        end
    end
    return color
end

function prevent_cycle!(v:: Int, w:: Int, x::Int, g, set, first_visit_to_tree, forbidden_colors,color)
    e = find(w, x, g, set)
    p, q = first_visit_to_tree[e]
    println(">>> first visit to tree : p = $p, q = $q")
    if p != v
        first_visit_to_tree[e] = (v,w)
    elseif q != w
        forbidden_colors[color[x]] = v
    end
end

function grow_star!(v, w,g,set,first_neighbor,color)
    make_set!(v,w,g,set)
    p, q = first_neighbor[color[w]]
    wc = color[w]
    println(">>> color of w = $wc")
    println(">>> first neighbor : p = $p, q = $q")
    if p != v
        first_neighbor[color[w]] = (v,w)
    else
        e1 = find(v,w,g,set)
        e2 = find(p,q,g,set)
        union!(set, e1, e2)
    end
end

function merge_trees!(v,w,x,g,set)
    e1 = find(v,w,g,set)
    e2 = find(w,x,g,set)
    if e1 != e2
        union!(set, e1, e2)
    end
end

function make_set!(v,w,g,set)
    edge_index = find_edge_index(v,w,g)
    push!(set,edge_index)
end

function min_index(forbidden_colors, v)
    return findfirst(!isequal(v), forbidden_colors)
end

function find(w, x, g, set)
    edge_index = find_edge_index(w, x, g)
    return find_root(set, edge_index)
end

function find_edge_index(v, w, g)
    #print("function called")
    pos = 1
    for i in edges(g)
        #print("inside loop")
        if (src(i)==v && dst(i)==w) || (src(i)==w && dst(i)==v)
            return pos
        end
        pos = pos + 1
    end
    throw(ArgumentError("$v and $w are not connected in the graph"))
end
