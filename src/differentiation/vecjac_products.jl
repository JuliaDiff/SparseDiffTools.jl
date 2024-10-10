function num_vecjac!(du, f::F, x, v, cache1 = similar(v), cache2 = similar(v),
        cache3 = similar(x); compute_f0 = true) where {F}
    compute_f0 && (f(cache1, x))
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    vv = reshape(v, size(cache1))
    cache3 .= x
    for i in 1:length(x)
        cache3[i] += ϵ
        f(cache2, cache3)
        cache3[i] = x[i]
        du[i] = (((cache2 .- cache1) ./ ϵ)' * vv)[1]
    end
    return du
end

# Special Non-Allocating case for StaticArrays
function num_vecjac(f::F, x::SArray, v::SArray, f0 = nothing) where {F}
    f0 === nothing ? (_f0 = f(x)) : (_f0 = f0)
    vv = reshape(v, axes(_f0))
    T = eltype(x)
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    du = zeros(typeof(x))
    for i in 1:length(x)
        cache = Base.setindex(x, x[i] + ϵ, i)
        f0 = f(cache)
        du = Base.setindex(du, (((f0 .- _f0) ./ ϵ)' * vv), i)
    end
    return du
end

function num_vecjac(f::F, x, v, f0 = nothing) where {F}
    f0 === nothing ? (_f0 = f(x)) : (_f0 = f0)
    vv = reshape(v, axes(_f0))
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    du = similar(x)
    cache = similar(x)
    copyto!(cache, x)
    for i in 1:length(x)
        cache = allowed_setindex!(cache, x[i] + ϵ, i)
        f0 = f(cache)
        cache = allowed_setindex!(cache, x[i], i)
        du = allowed_setindex!(du, (((f0 .- _f0) ./ ϵ)' * vv)[1], i)
    end
    return vec(du)
end

### Operator Forms

"""
    VecJac(f, u, [p, t]; fu = nothing, autodiff = AutoFiniteDiff())

Returns SciMLOperators.FunctionOperator which computes vector-jacobian product
`(df/du)ᵀ * v`.

!!! note

    For non-square jacobians with inplace `f`, `fu` must be specified, else `VecJac` assumes
    a square jacobian.

```julia
L = VecJac(f, u)

L * v         # = (df/du)ᵀ * v
mul!(w, L, v) # = (df/du)ᵀ * v

L(v, p, t; VJP_input = w)    # = (df/du)ᵀ * v
L(x, v, p, t; VJP_input = w) # = (df/du)ᵀ * v
```

## Allowed Function Signatures for `f`

For Out of Place Functions:

```julia
f(u, p, t)  # t !== nothing
f(u, p)     # p !== nothing
f(u)        # Otherwise
```

For In Place Functions:

```julia
f(du, u, p, t)  # t !== nothing
f(du, u, p)     # p !== nothing
f(du, u)        # Otherwise
```
"""
function VecJac(f, u::AbstractArray, p = nothing, t = nothing; fu = nothing,
        autodiff = AutoFiniteDiff(), use_deprecated_ordering::Val = Val(true), kwargs...)
    ff = JacFunctionWrapper(f, fu, u, p, t; use_deprecated_ordering)

    if !__internal_oop(ff) && autodiff isa AutoZygote
        msg = "Zygote requires an out of place method with signature f(u)."
        throw(ArgumentError(msg))
    end

    fu === nothing && (fu = __internal_oop(ff) ? ff(u) : u)

    op = _vecjac(ff, fu, u, autodiff)

    # NOTE: We pass `p`, `t` to Function Operator but we always use the cached version from
    #       JacFunctionWrapper
    return FunctionOperator(op, fu, u; p, t, isinplace = Val(true), outofplace = Val(true),
        islinear = true, accepted_kwargs = (:VJP_input,), kwargs...)
end

function _vecjac(f::F, fu, u, autodiff::AutoFiniteDiff) where {F}
    cache = (similar(fu), similar(fu), similar(u))
    pullback = nothing
    return AutoDiffVJP(f, u, cache, autodiff, pullback)
end

mutable struct AutoDiffVJP{AD, F, U, C, PB} <: AbstractAutoDiffVecProd
    """ Compute VJP of `f` at `u`, applied to vector `v`: `df/du' * u` """
    f::F
    """ input to `f` """
    u::U
    """ Cache for num_vecjac! when autodiff isa AutoFintieDiff """
    cache::C
    """ Type of automatic differentiation algorithm """
    autodiff::AD
    """ stores the result of Zygote.pullback for AutoZygote """
    pullback::PB
end

function update_coefficients(L::AutoDiffVJP{<:AutoFiniteDiff}, u, p, t; VJP_input = nothing)
    VJP_input !== nothing && (@set! L.u = VJP_input)
    @set! L.f = update_coefficients(L.f, L.u, p, t)
    return L
end

function update_coefficients!(L::AutoDiffVJP{<:AutoFiniteDiff}, u, p, t;
        VJP_input = nothing)
    VJP_input !== nothing && copy!(L.u, VJP_input)
    update_coefficients!(L.f, L.u, p, t)
    return L
end

# Interpret the call as df/du' * v
function (L::AutoDiffVJP{<:AutoFiniteDiff})(v, p, t; VJP_input = nothing)
    # ignore VJP_input as L.u was set in update_coefficients(...)
    return num_vecjac(L.f, L.u, v)
end

function (L::AutoDiffVJP{<:AutoFiniteDiff})(dv, v, p, t; VJP_input = nothing)
    # ignore VJP_input as L.u was set in update_coefficients!(...)
    return num_vecjac!(dv, L.f, L.u, v, L.cache...)
end

function Base.resize!(L::AutoDiffVJP, n::Integer)
    hasmethod(resize!, typeof((L.f, n))) && resize!(L.f, n)
    resize!(L.u, n)
    for v in L.cache
        resize!(v, n)
    end
end
