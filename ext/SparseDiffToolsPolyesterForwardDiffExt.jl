module SparseDiffToolsPolyesterForwardDiffExt

using ADTypes, SparseDiffTools, PolyesterForwardDiff
import ForwardDiff
import SparseDiffTools: AbstractMaybeSparseJacobianCache, AbstractMaybeSparsityDetection,
    ForwardColorJacCache, NoMatrixColoring, sparse_jacobian_cache, sparse_jacobian!,
    sparse_jacobian_static_array, __standard_tag, __chunksize

struct PolyesterForwardDiffJacobianCache{CO, CA, J, FX, X} <:
       AbstractMaybeSparseJacobianCache
    coloring::CO
    cache::CA
    jac_prototype::J
    fx::FX
    x::X
end

function sparse_jacobian_cache(ad::Union{AutoSparsePolyesterForwardDiff,
            AutoPolyesterForwardDiff}, sd::AbstractMaybeSparsityDetection, f::F, x;
        fx = nothing) where {F}
    coloring_result = sd(ad, f, x)
    fx = fx === nothing ? similar(f(x)) : fx
    if coloring_result isa NoMatrixColoring
        cache = __chunksize(ad, x)
        jac_prototype = nothing
    else
        @warn """Currently PolyesterForwardDiff does not support sparsity detection
                 natively. Falling back to using ForwardDiff.jl""" maxlog=1
        tag = __standard_tag(nothing, x)
        # Colored ForwardDiff passes `tag` directly into Dual so we need the `typeof`
        cache = ForwardColorJacCache(f, x, __chunksize(ad); coloring_result.colorvec,
            dx = fx, sparsity = coloring_result.jacobian_sparsity, tag = typeof(tag))
        jac_prototype = coloring_result.jacobian_sparsity
    end
    return PolyesterForwardDiffJacobianCache(coloring_result, cache, jac_prototype, fx, x)
end

function sparse_jacobian_cache(ad::Union{AutoSparsePolyesterForwardDiff,
            AutoPolyesterForwardDiff}, sd::AbstractMaybeSparsityDetection, f!::F, fx,
        x) where {F}
    coloring_result = sd(ad, f!, fx, x)
    if coloring_result isa NoMatrixColoring
        cache = __chunksize(ad, x)
        jac_prototype = nothing
    else
        @warn """Currently PolyesterForwardDiff does not support sparsity detection
                 natively. Falling back to using ForwardDiff.jl""" maxlog=1
        tag = __standard_tag(nothing, x)
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
        forwarddiff_color_jacobian(J, f, x, cache.cache) # Use Sparse ForwardDiff
    else
        PolyesterForwardDiff.threaded_jacobian!(f, J, x, cache.cache) # Don't try to exploit sparsity
    end
    return J
end

function sparse_jacobian!(J::AbstractMatrix, _, cache::PolyesterForwardDiffJacobianCache,
        f!::F, fx, x) where {F}
    if cache.cache isa ForwardColorJacCache
        forwarddiff_color_jacobian!(J, f!, x, cache.cache) # Use Sparse ForwardDiff
    else
        PolyesterForwardDiff.threaded_jacobian!(f!, fx, J, x, cache.cache) # Don't try to exploit sparsity
    end
    return J
end

end
