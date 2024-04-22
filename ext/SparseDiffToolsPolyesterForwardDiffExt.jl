module SparseDiffToolsPolyesterForwardDiffExt

using ADTypes, SparseDiffTools, PolyesterForwardDiff, UnPack, Random, SparseArrays
import ForwardDiff
import SparseDiffTools: AbstractMaybeSparseJacobianCache, AbstractMaybeSparsityDetection,
                        ForwardColorJacCache, NoMatrixColoring, sparse_jacobian_cache,
                        sparse_jacobian_cache_aux,
                        sparse_jacobian!,
                        sparse_jacobian_static_array, __standard_tag, __chunksize,
                        polyesterforwarddiff_color_jacobian

struct PolyesterForwardDiffJacobianCache{CO, CA, J, FX, X} <:
       AbstractMaybeSparseJacobianCache
    coloring::CO
    cache::CA
    jac_prototype::J
    fx::FX
    x::X
end

function sparse_jacobian_cache_aux(::ADTypes.ForwardMode,
        ad::Union{AutoSparse{<:AutoPolyesterForwardDiff}, AutoPolyesterForwardDiff},
        sd::AbstractMaybeSparsityDetection, f::F, x; fx = nothing) where {F}
    coloring_result = sd(ad, f, x)
    fx = fx === nothing ? similar(f(x)) : fx
    if coloring_result isa NoMatrixColoring
        cache = __chunksize(ad, x)
        jac_prototype = nothing
    else
        tag = __standard_tag(nothing, f, x)
        # Colored ForwardDiff passes `tag` directly into Dual so we need the `typeof`
        cache = ForwardColorJacCache(f, x, __chunksize(ad); coloring_result.colorvec,
            dx = fx, sparsity = coloring_result.jacobian_sparsity, tag = typeof(tag))
        jac_prototype = coloring_result.jacobian_sparsity
    end
    return PolyesterForwardDiffJacobianCache(coloring_result, cache, jac_prototype, fx, x)
end

function sparse_jacobian_cache_aux(::ADTypes.ForwardMode,
        ad::Union{AutoSparse{<:AutoPolyesterForwardDiff}, AutoPolyesterForwardDiff},
        sd::AbstractMaybeSparsityDetection, f!::F, fx, x) where {F}
    coloring_result = sd(ad, f!, fx, x)
    if coloring_result isa NoMatrixColoring
        cache = __chunksize(ad, x)
        jac_prototype = nothing
    else
        @warn """Currently PolyesterForwardDiff does not support sparsity detection
                 natively for inplace functions. Falling back to using
                ForwardDiff.jl""" maxlog=1
        tag = __standard_tag(nothing, f!, x)
        # Colored ForwardDiff passes `tag` directly into Dual so we need the `typeof`
        cache = ForwardColorJacCache(f!, x, __chunksize(ad); coloring_result.colorvec,
            dx = fx, sparsity = coloring_result.jacobian_sparsity, tag = typeof(tag))
        jac_prototype = coloring_result.jacobian_sparsity
    end
    return PolyesterForwardDiffJacobianCache(coloring_result, cache, jac_prototype, fx, x)
end

function sparse_jacobian!(J::AbstractMatrix, _, cache::PolyesterForwardDiffJacobianCache,
        f::F, x) where {F}
    if cache.cache isa ForwardColorJacCache
        polyesterforwarddiff_color_jacobian(J, f, x, cache.cache)
    else
        PolyesterForwardDiff.threaded_jacobian!(f, J, x, cache.cache) # Don't try to exploit sparsity
    end
    return J
end

function sparse_jacobian!(J::AbstractMatrix, _, cache::PolyesterForwardDiffJacobianCache,
        f!::F, fx, x) where {F}
    if cache.cache isa ForwardColorJacCache
        forwarddiff_color_jacobian!(J, f!, x, cache.cache)
    else
        PolyesterForwardDiff.threaded_jacobian!(f!, fx, J, x, cache.cache) # Don't try to exploit sparsity
    end
    return J
end

## Approximate Sparsity Detection
function (alg::ApproximateJacobianSparsity)(
        ad::AutoSparse{<:AutoPolyesterForwardDiff}, f::F, x; fx = nothing, kwargs...) where {F}
    @unpack ntrials, rng = alg
    fx = fx === nothing ? f(x) : fx
    ck = __chunksize(ad, x)
    J = fill!(similar(fx, length(fx), length(x)), 0)
    J_cache = similar(J)
    x_ = similar(x)
    for _ in 1:ntrials
        randn!(rng, x_)
        PolyesterForwardDiff.threaded_jacobian!(f, J_cache, x_, ck)
        @. J += abs(J_cache)
    end
    return (JacPrototypeSparsityDetection(; jac_prototype = sparse(J), alg.alg))(ad, f, x;
        fx, kwargs...)
end

function (alg::ApproximateJacobianSparsity)(
        ad::AutoSparse{<:AutoPolyesterForwardDiff}, f::F, fx, x;
        kwargs...) where {F}
    @unpack ntrials, rng = alg
    ck = __chunksize(ad, x)
    J = fill!(similar(fx, length(fx), length(x)), 0)
    J_cache = similar(J)
    x_ = similar(x)
    for _ in 1:ntrials
        randn!(rng, x_)
        PolyesterForwardDiff.threaded_jacobian!(f, fx, J_cache, x_, ck)
        @. J += abs(J_cache)
    end
    return (JacPrototypeSparsityDetection(; jac_prototype = sparse(J), alg.alg))(ad, f, x;
        fx, kwargs...)
end

end
