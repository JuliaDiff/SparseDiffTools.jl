struct ForwardDiffJacobianCache{CO, CA, J, FX, X} <: AbstractMaybeSparseJacobianCache
    coloring::CO
    cache::CA
    jac_prototype::J
    fx::FX
    x::X
end

struct SparseDiffToolsTag end

function sparse_jacobian_cache(ad::Union{AutoSparseForwardDiff, AutoForwardDiff},
        sd::AbstractMaybeSparsityDetection, f::F, x; fx = nothing) where {F}
    coloring_result = sd(ad, f, x)
    fx = fx === nothing ? similar(f(x)) : fx
    if coloring_result isa NoMatrixColoring
        cache = ForwardDiff.JacobianConfig(f, x, __chunksize(ad, x),
            ifelse(ad.tag === nothing, SparseDiffToolsTag(), ad.tag))
        jac_prototype = nothing
    else
        cache = ForwardColorJacCache(f, x, __chunksize(ad); coloring_result.colorvec,
            dx = fx, sparsity = coloring_result.jacobian_sparsity, ad.tag)
        jac_prototype = coloring_result.jacobian_sparsity
    end
    return ForwardDiffJacobianCache(coloring_result, cache, jac_prototype, fx, x)
end

function sparse_jacobian_cache(ad::Union{AutoSparseForwardDiff, AutoForwardDiff},
        sd::AbstractMaybeSparsityDetection, f!::F, fx, x) where {F}
    coloring_result = sd(ad, f!, fx, x)
    if coloring_result isa NoMatrixColoring
        cache = ForwardDiff.JacobianConfig(f!, fx, x, __chunksize(ad, x),
            ifelse(ad.tag === nothing, SparseDiffToolsTag(), ad.tag))
        jac_prototype = nothing
    else
        cache = ForwardColorJacCache(f!, x, __chunksize(ad); coloring_result.colorvec,
            dx = fx, sparsity = coloring_result.jacobian_sparsity, ad.tag)
        jac_prototype = coloring_result.jacobian_sparsity
    end
    return ForwardDiffJacobianCache(coloring_result, cache, jac_prototype, fx, x)
end

function sparse_jacobian!(J::AbstractMatrix, _, cache::ForwardDiffJacobianCache, f::F,
        x) where {F}
    if cache.cache isa ForwardColorJacCache
        forwarddiff_color_jacobian(J, f, x, cache.cache) # Use Sparse ForwardDiff
    else
        ForwardDiff.jacobian!(J, f, x, cache.cache) # Don't try to exploit sparsity
    end
    return J
end

function sparse_jacobian!(J::AbstractMatrix, _, cache::ForwardDiffJacobianCache, f!::F, fx,
        x) where {F}
    if cache.cache isa ForwardColorJacCache
        forwarddiff_color_jacobian!(J, f!, x, cache.cache) # Use Sparse ForwardDiff
    else
        ForwardDiff.jacobian!(J, f!, fx, x, cache.cache) # Don't try to exploit sparsity
    end
    return J
end
