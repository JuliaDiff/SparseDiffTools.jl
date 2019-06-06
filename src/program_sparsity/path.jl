#### Path

# First just do it for the case where there we assume
# tainted gotoifnots do not go in a loop!
# TODO: write a thing to detect this! (overdub predicates only in tainted ifs)
# implement snapshotting function state as an optimization for branch exploration
mutable struct Path
    path::BitVector
    cursor::Int
end

Path() = Path([], 1)

function increment!(bitvec)
    for i=1:length(bitvec)
        if bitvec[i] === true
            bitvec[i] = false
        else
            bitvec[i] = true
            break
        end
    end
end

function reset!(p::Path)
    p.cursor=1
    increment!(p.path)
    nothing
end

function alldone(p::Path) # must be called at the end of the function!
    all(identity, p.path)
end

function this_here_predicate!(p::Path)
    if p.cursor > length(p.path)
        push!(p.path, false)
    else
        p.path[p.cursor]
    end
    val = p.path[p.cursor]
    p.cursor+=1
    val
end

alldone(c::SparsityContext) = alldone(c.metadata[2])
reset!(c::SparsityContext) = reset!(c.metadata[2])
this_here_predicate!(c::SparsityContext) = this_here_predicate!(c.metadata[2])

#=
julia> p=Path()
Path(Bool[], 1)

julia> alldone(p) # must be called at the end of a full run
true

julia> this_here_predicate!(p)
false

julia> alldone(p) # must be called at the end of a full run
false

julia> this_here_predicate!(p)
false

julia> p
Path(Bool[false, false], 3)

julia> alldone(p)
false

julia> reset!(p)

julia> p
Path(Bool[true, false], 1)

julia> this_here_predicate!(p)
true

julia> this_here_predicate!(p)
false

julia> alldone(p)
false

julia> reset!(p)

julia> p
Path(Bool[false, true], 1)

julia> this_here_predicate!(p)
false

julia> this_here_predicate!(p)
true

julia> reset!(p)

julia> this_here_predicate!(p)
true

julia> this_here_predicate!(p)
true

julia> alldone(p)
true
=#
