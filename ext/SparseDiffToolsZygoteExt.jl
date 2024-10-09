module SparseDiffToolsZygoteExt

using ADTypes, LinearAlgebra, Zygote
import SparseDiffTools: SparseDiffTools, DeivVecTag, AutoDiffVJP, __test_backend_loaded
import ForwardDiff: ForwardDiff, Dual, partials
import SciMLOperators: update_coefficients, update_coefficients!
import Setfield: @set!

import SparseDiffTools: numback_hesvec!,
                        numback_hesvec, autoback_hesvec!, autoback_hesvec, auto_vecjac!,
                        auto_vecjac
import SparseDiffTools: __f̂, __jacobian!, __gradient, __gradient!
import ADTypes: AutoZygote, AutoSparse

@inline __test_backend_loaded(::Union{AutoSparse{<:AutoZygote}, AutoZygote}) = nothing

## Satisfying High-Level Interface for Sparse Jacobians
function __gradient(::Union{AutoSparse{<:AutoZygote}, AutoZygote}, f::F, x, cols) where {F}
    _, ∂x, _ = Zygote.gradient(__f̂, f, x, cols)
    return vec(∂x)
end

function __gradient!(
        ::Union{AutoSparse{<:AutoZygote}, AutoZygote}, f!::F, fx, x, cols) where {F}
    return error("Zygote.jl cannot differentiate in-place (mutating) functions.")
end

# Zygote doesn't provide a way to accumulate directly into `J`. So we modify the code from
# https://github.com/FluxML/Zygote.jl/blob/82c7a000bae7fb0999275e62cc53ddb61aed94c7/src/lib/grad.jl#L140-L157C4
import Zygote: _jvec, _eyelike, _gradcopy!

@views function __jacobian!(
        J::AbstractMatrix, ::Union{AutoSparse{<:AutoZygote}, AutoZygote}, f::F,
        x) where {F}
    y, back = Zygote.pullback(_jvec ∘ f, x)
    δ = _eyelike(y)
    for k in LinearIndices(y)
        grad = only(back(δ[:, k]))
        _gradcopy!(J[k, :], grad)
    end
    return J
end

function __jacobian!(
        _, ::Union{AutoSparse{<:AutoZygote}, AutoZygote}, f!::F, fx, x) where {F}
    return error("Zygote.jl cannot differentiate in-place (mutating) functions.")
end

### Jac, Hes products

function numback_hesvec!(dy, f::F, x, v, cache1 = similar(v), cache2 = similar(v),
        cache3 = similar(v)) where {F}
    g = let f = f
        (dx, x) -> dx .= first(Zygote.gradient(f, x))
    end
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    @. cache3 = x + ϵ * v
    g(cache1, cache3)
    @. cache3 = x - ϵ * v
    g(cache2, cache3)
    @. dy = (cache1 - cache2) / (2ϵ)
end

function numback_hesvec(f::F, x, v) where {F}
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    x += ϵ * v
    gxp = first(Zygote.gradient(f, x))
    x -= 2ϵ * v
    gxm = first(Zygote.gradient(f, x))
    (gxp - gxm) / (2ϵ)
end

@inline function _default_autoback_hesvec_cache(x, v)
    T = typeof(ForwardDiff.Tag(DeivVecTag(), eltype(x)))
    return Dual{T, eltype(x), 1}.(x, ForwardDiff.Partials.(tuple.(reshape(v, size(x)))))
end

function autoback_hesvec!(dy, f, x, v, cache1 = _default_autoback_hesvec_cache(x, v),
        cache2 = _default_autoback_hesvec_cache(x, v))
    g = let f = f
        (dx, x) -> dx .= first(Zygote.gradient(f, x))
    end
    # Reset each dual number in cache1 to primal = dual = 1.
    cache1 .= eltype(cache1).(x, ForwardDiff.Partials.(tuple.(reshape(v, size(x)))))
    g(cache2, cache1)
    dy .= partials.(cache2, 1)
end

function autoback_hesvec(f, x, v)
    g = x -> first(Zygote.gradient(f, x))
    y = _default_autoback_hesvec_cache(x, v)
    return ForwardDiff.partials.(g(y), 1)
end

## VecJac products

# VJP methods
function auto_vecjac!(du, f::F, x, v) where {F}
    !hasmethod(f, typeof((x,))) &&
        error("For inplace function use autodiff = AutoFiniteDiff()")
    du .= reshape(SparseDiffTools.auto_vecjac(f, x, v), size(du))
end

function auto_vecjac(f::F, x, v) where {F}
    y, back = Zygote.pullback(f, x)
    return vec(only(back(reshape(v, size(y)))))
end

# overload operator interface
function SparseDiffTools._vecjac(f::F, _, u, autodiff::AutoZygote) where {F}
    !hasmethod(f, typeof((u,))) &&
        error("For inplace function use autodiff = AutoFiniteDiff()")
    pullback = Zygote.pullback(f, u)
    return AutoDiffVJP(f, u, (), autodiff, pullback)
end

function update_coefficients(L::AutoDiffVJP{<:AutoZygote}, u, p, t; VJP_input = nothing)
    VJP_input !== nothing && (@set! L.u = VJP_input)
    @set! L.f = update_coefficients(L.f, L.u, p, t)
    @set! L.pullback = Zygote.pullback(L.f, L.u)
    return L
end

function update_coefficients!(L::AutoDiffVJP{<:AutoZygote}, u, p, t; VJP_input = nothing)
    VJP_input !== nothing && copy!(L.u, VJP_input)
    update_coefficients!(L.f, L.u, p, t)
    L.pullback = Zygote.pullback(L.f, L.u)
    return L
end

# Interpret the call as df/du' * v
function (L::AutoDiffVJP{<:AutoZygote})(v, p, t; VJP_input = nothing)
    # ignore VJP_input as pullback was computed in update_coefficients(...)
    y, back = L.pullback
    return vec(only(back(reshape(v, size(y)))))
end

# prefer non in-place method
function (L::AutoDiffVJP{<:AutoZygote})(dv, v, p, t; VJP_input = nothing)
    # ignore VJP_input as pullback was computed in update_coefficients!(...)
    _dv = L(v, p, t; VJP_input)
    copy!(dv, _dv)
end

end # module
