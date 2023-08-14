function num_vecjac!(du, f, x, v, cache1 = similar(v), cache2 = similar(v);
    compute_f0 = true)
    compute_f0 && (f(cache1, x))
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    vv = reshape(v, size(x))
    for i in 1:length(x)
        x[i] += ϵ
        f(cache2, x)
        x[i] -= ϵ
        du[i] = (((cache2 .- cache1) ./ ϵ)' * vv)[1]
    end
    return du
end

function num_vecjac(f, x, v, f0 = nothing)
    vv = reshape(v, axes(x))
    f0 === nothing ? _f0 = f(x) : _f0 = f0
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
    VecJac(f, u, [p, t]; autodiff = AutoFiniteDiff())

Returns SciMLOperators.FunctionOperator which computes vector-jacobian
product `df/du * v`.

```
L = VecJac(f, u)

L * v         # = df/du * v
mul!(w, L, v) # = df/du * v

L(v, p, t; VJP_input = w)    # = df/dw * v
L(x, v, p, t; VJP_input = w) # = df/dw * v
```
"""
function VecJac(f, u::AbstractArray, p = nothing, t = nothing;
    autodiff = AutoFiniteDiff(), kwargs...)
    L = _vecjac(f, u, autodiff)
    IIP, OOP = get_iip_oop(L)

    if isa(autodiff, AutoZygote) & !OOP
        msg = "Zygote requires an out of place method with signature f(u)."
        throw(ArgumentError(msg))
    end

    return FunctionOperator(L, u, u; isinplace = IIP, outofplace = OOP,
        p, t, islinear = true, accepted_kwargs = (:VJP_input,), kwargs...)
end

function _vecjac(f, u, autodiff::AutoFiniteDiff)
    cache = (similar(u), similar(u))
    pullback = nothing

    AutoDiffVJP(f, u, cache, autodiff, pullback)
end

mutable struct AutoDiffVJP{AD, IIP, OOP, F, U, C, PB} <: AbstractAutoDiffVecProd
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

    function AutoDiffVJP(f, u, cache, autodiff, pullback)
        outofplace = static_hasmethod(f, typeof((u,)))
        isinplace = static_hasmethod(f, typeof((u, u)))

        if !(isinplace) & !(outofplace)
            msg = "$f must have signature f(u), or f(du, u)"
            throw(ArgumentError(msg))
        end

        new{
            typeof(autodiff),
            isinplace,
            outofplace,
            typeof(f),
            typeof(u),
            typeof(cache),
            typeof(pullback),
        }(f,
            u,
            cache,
            autodiff,
            pullback)
    end
end

function get_iip_oop(::AutoDiffVJP{AD, IIP, OOP}) where {AD, IIP, OOP}
    IIP, OOP
end

function update_coefficients(L::AutoDiffVJP{AD}, u, p, t;
    VJP_input = nothing) where {AD <: AutoFiniteDiff}
    if !isnothing(VJP_input)
        @set! L.u = VJP_input
    end

    @set! L.f = update_coefficients(L.f, L.u, p, t)
end

function update_coefficients!(L::AutoDiffVJP{AD}, u, p, t;
    VJP_input = nothing) where {AD <: AutoFiniteDiff}
    if !isnothing(VJP_input)
        copy!(L.u, VJP_input)
    end

    update_coefficients!(L.f, L.u, p, t)

    L
end

# Interpret the call as df/du' * v
function (L::AutoDiffVJP{AD})(v, p, t; VJP_input = nothing) where {AD <: AutoFiniteDiff}
    # ignore VJP_input as L.u was set in update_coefficients(...)
    num_vecjac(L.f, L.u, v)
end

function (L::AutoDiffVJP{AD})(dv, v, p, t; VJP_input = nothing) where {AD <: AutoFiniteDiff}
    # ignore VJP_input as L.u was set in update_coefficients!(...)
    num_vecjac!(dv, L.f, L.u, v, L.cache...)
end

function Base.resize!(L::AutoDiffVJP, n::Integer)
    static_hasmethod(resize!, typeof((L.f, n))) && resize!(L.f, n)
    resize!(L.u, n)

    for v in L.cache
        resize!(v, n)
    end
end
#
