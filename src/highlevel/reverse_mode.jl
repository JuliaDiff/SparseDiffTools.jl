struct ReverseModeJacobianCache{CO, CA, J, FX, X, I} <: AbstractMaybeSparseJacobianCache
    coloring::CO
    cache::CA
    jac_prototype::J
    fx::FX
    x::X
    idx_vec::I
end

__getfield(c::ReverseModeJacobianCache, ::Val{:jac_prototype}) = c.jac_prototype

function sparse_jacobian_cache_aux(::ReverseMode, ad::AbstractADType,
        sd::AbstractMaybeSparsityDetection, f::F, x; fx = nothing) where {F}
    fx = fx === nothing ? similar(f(x)) : fx
    coloring_result = sd(ad, f, x)
    jac_prototype = __getfield(coloring_result, Val(:jacobian_sparsity))
    return ReverseModeJacobianCache(coloring_result, nothing, jac_prototype, fx, x,
        collect(1:length(fx)))
end

function sparse_jacobian_cache_aux(::ReverseMode, ad::AbstractADType,
        sd::AbstractMaybeSparsityDetection, f!::F, fx, x) where {F}
    coloring_result = sd(ad, f!, fx, x)
    jac_prototype = __getfield(coloring_result, Val(:jacobian_sparsity))
    return ReverseModeJacobianCache(coloring_result, nothing, jac_prototype, fx, x,
        collect(1:length(fx)))
end

function sparse_jacobian!(J::AbstractMatrix, ad, cache::ReverseModeJacobianCache, args...)
    if cache.coloring isa NoMatrixColoring
        __test_backend_loaded(ad)
        return __jacobian!(J, ad, args...)
    else
        return __sparse_jacobian_reverse_impl!(J, ad, cache.idx_vec, cache.coloring,
            args...)
    end
end

function __sparse_jacobian_reverse_impl!(J::AbstractMatrix, ad, idx_vec,
        cache::MatrixColoringResult, f::F, x) where {F}
    return __sparse_jacobian_reverse_impl!(J, ad, idx_vec, cache, f, nothing, x)
end

function __sparse_jacobian_reverse_impl!(J::AbstractMatrix, ad, idx_vec,
        cache::MatrixColoringResult, f::F, fx, x) where {F}
    # If `fx` is `nothing` then assume `f` is not in-place
    __test_backend_loaded(ad)
    x_ = __maybe_copy_x(ad, x)
    fx_ = __maybe_copy_x(ad, fx)
    @unpack colorvec, nz_rows, nz_cols = cache
    for c in 1:maximum(colorvec)
        @. idx_vec = colorvec == c
        if fx === nothing
            gs = __gradient(ad, f, x_, idx_vec)
        else
            gs = __gradient!(ad, f, fx_, x_, idx_vec)
        end
        pick_idxs = filter(i -> colorvec[nz_rows[i]] == c, 1:length(nz_rows))
        row_idxs = nz_rows[pick_idxs]
        col_idxs = nz_cols[pick_idxs]
        len_cols = length(col_idxs)
        unused_cols = setdiff(1:size(J, 2), col_idxs)
        perm_cols = sortperm(vcat(col_idxs, unused_cols))
        row_idxs = vcat(row_idxs, zeros(Int, size(J, 2) - len_cols))[perm_cols]
        # FIXME: Assumes fast scalar indexing currently. Very easy to write a kernel to do
        #        this in parallel using KA.jl.
        for i in axes(J, 1), j in axes(J, 2)
            i == row_idxs[j] && (J[i, j] = gs[j])
        end
    end
    return J
end
