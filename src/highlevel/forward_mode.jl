struct ForwardDiffJacobianCache{CO, CA, J, FX, X} <: AbstractMaybeSparseJacobianCache
    coloring::CO
    cache::CA
    jac_prototype::J
    fx::FX
    x::X
end

__getfield(c::ForwardDiffJacobianCache, ::Val{:jac_prototype}) = c.jac_prototype

__standard_tag(::Nothing, f::F, x) where {F} = ForwardDiff.Tag(f, eltype(x))
__standard_tag(tag::ForwardDiff.Tag, ::F, _) where {F} = tag
__standard_tag(tag, f::F, x) where {F} = ForwardDiff.Tag(f, eltype(x))

function sparse_jacobian_cache_aux(
        ::ForwardMode, ad::Union{AutoSparse{<:AutoForwardDiff}, AutoForwardDiff},
        sd::AbstractMaybeSparsityDetection, f::F, x; fx = nothing) where {F}
    coloring_result = sd(ad, f, x)
    fx = fx === nothing ? similar(f(x)) : fx
    tag = __standard_tag(my_dense_ad(ad).tag, f, x)
    if coloring_result isa NoMatrixColoring
        cache = ForwardDiff.JacobianConfig(f, x, __chunksize(ad, x), tag)
        jac_prototype = nothing
    else
        # Colored ForwardDiff passes `tag` directly into Dual so we need the `typeof`
        cache = ForwardColorJacCache(f, x, __chunksize(ad); coloring_result.colorvec,
            dx = fx, sparsity = coloring_result.jacobian_sparsity, tag = typeof(tag))
        jac_prototype = coloring_result.jacobian_sparsity
    end
    return ForwardDiffJacobianCache(coloring_result, cache, jac_prototype, fx, x)
end

function sparse_jacobian_cache_aux(
        ::ForwardMode, ad::Union{AutoSparse{<:AutoForwardDiff}, AutoForwardDiff},
        sd::AbstractMaybeSparsityDetection, f!::F, fx, x) where {F}
    coloring_result = sd(ad, f!, fx, x)
    tag = __standard_tag(my_dense_ad(ad).tag, f!, x)
    if coloring_result isa NoMatrixColoring
        cache = ForwardDiff.JacobianConfig(f!, fx, x, __chunksize(ad, x), tag)
        jac_prototype = nothing
    else
        # Colored ForwardDiff passes `tag` directly into Dual so we need the `typeof`
        cache = ForwardColorJacCache(f!, x, __chunksize(ad); coloring_result.colorvec,
            dx = fx, sparsity = coloring_result.jacobian_sparsity, tag = typeof(tag))
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

function sparse_jacobian_static_array(_, cache::ForwardDiffJacobianCache, f, x::SArray)
    if cache.cache isa ForwardColorJacCache
        return forwarddiff_color_jacobian(f, x, cache.cache)
    else
        return ForwardDiff.jacobian(f, x, cache.cache)
    end
end
