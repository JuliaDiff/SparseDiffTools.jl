struct TermCombination
    terms::Set{Dict{Int, Int}} # idx => pow
end

@eval Base.zero(::Type{TermCombination}) = $(TermCombination(Set{Dict{Int,Int}}()))
@eval Base.one(::Type{TermCombination}) = $(TermCombination(Set([Dict{Int,Int}()])))

function Base.:(==)(comb1::TermCombination, comb2::TermCombination)
    comb1.terms == comb2.terms && return true

    n1 = reduce(max, (k for (k,_) in Iterators.flatten(comb1.terms)), init=0)
    n2 = reduce(max, (k for (k,_) in Iterators.flatten(comb2.terms)), init=0)
    n = max(n1, n2)

    _sparse(comb1, n) == _sparse(comb2, n)
end

function Base.:+(comb1::TermCombination, comb2::TermCombination)
    if isone(comb1) && !iszero(comb2)
        return comb2
    elseif isone(comb2) && !iszero(comb1)
        return comb1
    elseif comb1 === comb2
        return comb1
    end
    TermCombination(union(comb1.terms, comb2.terms))
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
    if isone(comb1)
        return comb2
    elseif isone(comb2)
        return comb1
    elseif comb1 === comb2 # squaring optimization
        terms = comb1.terms
        # turns out it's enough to track
        # a^2*b^2
        # and a^2 + b^2 + ab
        # have the same hessian sparsity
        t = Dict(k=>2 for (k,_) in
                 Iterators.flatten(terms))
        TermCombination(Set([t]))
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
        TermCombination(union(t1, t2))
        =#
    else
        Set([_merge(dict1, dict2)
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
