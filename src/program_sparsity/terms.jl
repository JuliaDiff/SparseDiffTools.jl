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
        d[k] = min(2, get(dict1, k, 0) + v)
    end
    d
end

function Base.:*(comb1::TermCombination, comb2::TermCombination)
    if comb1 === comb2 # squaring optimization
        terms = comb1.terms
        # turns out it's enough to track
        # a^2*b^2
        # and a^2 + b^2 + ab
        # have the same hessian sparsity
        t = Dict(k=>2 for (k,_) in
                 Iterators.flatten(terms))
        TermCombination([t])
        #=
        # square each term
        t1 = [Dict(k=>2 for (k,_) in dict)
              for dict in comb1.terms]
        # multiply each term
        t2 = Dict{Int,Int}[]
        for i in 1:length(terms)
            for j in i+1:length(terms)
                push!(t2, _merge(terms[i], terms[j]))
            end
        end
        TermCombination(vcat(t1, t2))
        =#
    else
        vec([_merge(dict1, dict2)
               for dict1 in comb1.terms,
               dict2 in comb2.terms]) |> TermCombination
    end
end
Base.:*(comb1::TermCombination) = comb1
Base.iszero(c::TermCombination) = isempty(c.terms)
Base.isone(c::TermCombination) = all(isempty, c.terms)

function _sparse(t::TermCombination, n)
    I = Int[]
    J = Int[]
    for dict in t.terms
        kv = collect(pairs(dict))
        for i in 1:length(kv)
            k, v = kv[i]
            if v>=2
                push!(I, k)
                push!(J, k)
            end
            for j in i+1:length(kv)
                if v >= 1 && kv[j][2] >= 1
                    push!(I, k)
                    push!(J, kv[j][1])
                end
            end
        end
    end
    s1 = sparse(I,J,fill!(BitVector(undef, length(I)), true),n,n)
    s1 .| s1'
end
