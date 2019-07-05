struct TermCombination
    terms::Vector{Dict{Int, Int}} # idx => pow
end

Base.zero(::Type{TermCombination}) = TermCombination([])
Base.one(::Type{TermCombination}) = TermCombination([Dict{Int,Int}()])

function Base.:+(comb1::TermCombination, comb2::TermCombination)
    TermCombination(vcat(comb1.terms, comb2.terms))
end
Base.:+(comb1::TermCombination) = comb1

function _merge(dict1, dict2)
    d = copy(dict1)
    for (k, v) in dict2
        d[k] = get(dict1, k, 0) + v
    end
    d
end

function Base.:*(comb1::TermCombination, comb2::TermCombination)
    vec([_merge(dict1, dict2)
           for dict1 in comb1.terms,
           dict2 in comb2.terms]) |> TermCombination
end
Base.:*(comb1::TermCombination) = comb1

