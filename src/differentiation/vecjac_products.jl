function num_vecjac!(du, f::F, x, v, cache1 = similar(v), cache2 = similar(v);
        compute_f0 = true) where {F}
    compute_f0 && (f(cache1, x))
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    vv = reshape(v, size(cache1))
    for i in 1:length(x)
        x[i] += ϵ
        f(cache2, x)
        x[i] -= ϵ
        du[i] = (((cache2 .- cache1) ./ ϵ)' * vv)[1]
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
    for i in 1:length(x)
        x[i] += ϵ
        f0 = f(x)
        x[i] -= ϵ
        du[i] = (((f0 .- _f0) ./ ϵ)' * vv)[1]
    end
    return vec(du)
end

### Operator Forms

"""
    VecJac(f, u, [p, t]; fu = nothing, autodiff = AutoFiniteDiff())

Returns SciMLOperators.FunctionOperator which computes vector-jacobian product `df/du * v`.

!!! note

    For non-square jacobians with inplace `f`, `fu` must be specified, else `VecJac` assumes
    a square jacobian.

```julia
L = VecJac(f, u)

L * v         # = df/du * v
mul!(w, L, v) # = df/du * v

L(v, p, t; VJP_input = w)    # = df/dw * v
L(x, v, p, t; VJP_input = w) # = df/dw * v
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
        autodiff = AutoFiniteDiff(), kwargs...)
    ff = VecJacFunctionWrapper(f, fu, u, p, t)

    if !__internal_oop(ff) && autodiff isa AutoZygote
        msg = "Zygote requires an out of place method with signature f(u)."
        throw(ArgumentError(msg))
    end

    fu === nothing && (fu = __internal_oop(ff) ? ff(u) : u)

    op = _vecjac(ff, fu, u, autodiff)

    # FIXME: FunctionOperator is terribly type unstable. It makes it `::Any`
    # NOTE: We pass `p`, `t` to Function Operator but we always use the cached version from
    #       VecJacFunctionWrapper
    return FunctionOperator(op, fu, u; p, t, isinplace = true, outofplace = true,
        islinear = true, accepted_kwargs = (:VJP_input,), kwargs...)
end

mutable struct VecJacFunctionWrapper{iip, oop, mode, F, FU, P, T} <: Function
    f::F
    fu::FU
    p::P
    t::T
end

function SciMLOperators.update_coefficients!(L::VecJacFunctionWrapper{iip, oop, mode}, _,
        p, t) where {iip, oop, mode}
    mode == 1 && (L.t = t)
    mode == 2 && (L.p = p)
    return L
end
function SciMLOperators.update_coefficients(L::VecJacFunctionWrapper{iip, oop, mode}, _, p,
        t) where {iip, oop, mode}
    return VecJacFunctionWrapper{iip, oop, mode, typeof(L.f), typeof(L.fu), typeof(p),
        typeof(t)}(L.f, L.fu, p,
        t)
end

__internal_iip(::VecJacFunctionWrapper{iip}) where {iip} = iip
__internal_oop(::VecJacFunctionWrapper{iip, oop}) where {iip, oop} = oop

(f::VecJacFunctionWrapper{true, oop, 1})(fu, u) where {oop} = f.f(fu, u, f.p, f.t)
(f::VecJacFunctionWrapper{true, oop, 2})(fu, u) where {oop} = f.f(fu, u, f.p)
(f::VecJacFunctionWrapper{true, oop, 3})(fu, u) where {oop} = f.f(fu, u)
(f::VecJacFunctionWrapper{true, true, 1})(u) = f.f(u, f.p, f.t)
(f::VecJacFunctionWrapper{true, true, 2})(u) = f.f(u, f.p)
(f::VecJacFunctionWrapper{true, true, 3})(u) = f.f(u)
(f::VecJacFunctionWrapper{true, false, 1})(u) = (f.f(f.fu, u, f.p, f.t); copy(f.fu))
(f::VecJacFunctionWrapper{true, false, 2})(u) = (f.f(f.fu, u, f.p); copy(f.fu))
(f::VecJacFunctionWrapper{true, false, 3})(u) = (f.f(f.fu, u); copy(f.fu))

(f::VecJacFunctionWrapper{false, true, 1})(fu, u) = (vec(fu) .= vec(f.f(u, f.p, f.t)))
(f::VecJacFunctionWrapper{false, true, 2})(fu, u) = (vec(fu) .= vec(f.f(u, f.p)))
(f::VecJacFunctionWrapper{false, true, 3})(fu, u) = (vec(fu) .= vec(f.f(u)))
(f::VecJacFunctionWrapper{false, true, 1})(u) = f.f(u, f.p, f.t)
(f::VecJacFunctionWrapper{false, true, 2})(u) = f.f(u, f.p)
(f::VecJacFunctionWrapper{false, true, 3})(u) = f.f(u)

function VecJacFunctionWrapper(f::F, fu_, u, p, t) where {F}
    fu = fu_ === nothing ? copy(u) : copy(fu_)
    if t !== nothing
        iip = static_hasmethod(f, typeof((fu, u, p, t)))
        oop = static_hasmethod(f, typeof((u, p, t)))
        if !iip && !oop
            throw(ArgumentError("`f(u, p, t)` or `f(fu, u, p, t)` not defined for `f`"))
        end
        return VecJacFunctionWrapper{iip, oop, 1, F, typeof(fu), typeof(p), typeof(t)}(f,
            fu, p, t)
    elseif p !== nothing
        iip = static_hasmethod(f, typeof((fu, u, p)))
        oop = static_hasmethod(f, typeof((u, p)))
        if !iip && !oop
            throw(ArgumentError("`f(u, p)` or `f(fu, u, p)` not defined for `f`"))
        end
        return VecJacFunctionWrapper{iip, oop, 2, F, typeof(fu), typeof(p), typeof(t)}(f,
            fu, p, t)
    else
        iip = static_hasmethod(f, typeof((fu, u)))
        oop = static_hasmethod(f, typeof((u,)))
        if !iip && !oop
            throw(ArgumentError("`f(u)` or `f(fu, u)` not defined for `f`"))
        end
        return VecJacFunctionWrapper{iip, oop, 3, F, typeof(fu), typeof(p), typeof(t)}(f,
            fu, p, t)
    end
end

function _vecjac(f::F, fu, u, autodiff::AutoFiniteDiff) where {F}
    cache = (similar(fu), similar(fu))
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
    static_hasmethod(resize!, typeof((L.f, n))) && resize!(L.f, n)
    resize!(L.u, n)
    for v in L.cache
        resize!(v, n)
    end
end
