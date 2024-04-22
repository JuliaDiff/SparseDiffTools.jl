function sparse_jacobian_cache_aux(::ForwardOrReverseMode, ad::AbstractADType,
        sd::AbstractMaybeSparsityDetection, f::F, x; fx = nothing) where {F}
    if ad isa Union{AutoEnzyme, AutoSparse{<:AutoEnzyme}}
        return sparse_jacobian_cache_aux(ReverseMode(), ad, sd, f, x; fx)
    elseif ad isa Union{AutoDiffractor, AutoSparse{<:AutoDiffractor}}
        return sparse_jacobian_cache_aux(ForwardMode(), ad, sd, f, x; fx)
    else
        error("Unknown mixed mode AD")
    end
end

function sparse_jacobian_cache_aux(::ForwardOrReverseMode, ad::AbstractADType,
        sd::AbstractMaybeSparsityDetection, f!::F, fx, x) where {F}
    if ad isa Union{AutoEnzyme, AutoSparse{<:AutoEnzyme}}
        return sparse_jacobian_cache_aux(ReverseMode(), ad, sd, f!, fx, x)
    elseif ad isa Union{AutoDiffractor, AutoSparse{<:AutoDiffractor}}
        return sparse_jacobian_cache_aux(ForwardMode(), ad, sd, f!, fx, x)
    else
        error("Unknown mixed mode AD")
    end
end
