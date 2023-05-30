function num_vecjac!(du,
                     f,
                     x,
                     v,
                     cache1 = similar(v),
                     cache2 = similar(v);
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
"""
function VecJac(f, u::AbstractArray, p = nothing, t = nothing;
                autodiff = AutoFiniteDiff(), kwargs...)

    L = _vecjac(f, u, autodiff)
    IIP, OOP = get_iip_oop(L)

    FunctionOperator(L, u, u; isinplace = IIP, outofplace = OOP,
                     p = p, t = t, islinear = true, kwargs...)
end

function _vecjac(f, u, autodiff::AutoFiniteDiff)

    cache = (similar(u), similar(u))
    pullback = nothing

    AutoDiffVJP(f, u, cache, autodiff, pullback)
end

mutable struct AutoDiffVJP{AD, IIP, OOP, F, U, C, PB} <: AbstractAutoDiffVecProd
    f::F
    u::U
    cache::C
    autodiff::AD
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
           }(
             f, u, cache, autodiff, pullback,
            )
    end
end

function get_iip_oop(::AutoDiffVJP{AD, IIP, OOP}) where{AD, IIP, OOP}
    IIP, OOP
end

function update_coefficients(L::AutoDiffVJP{AD}, u, p, t) where{AD <: AutoFiniteDiff}
    @set! L.f = update_coefficients(L.f, u, p, t)
    @set! L.u = u
end

function update_coefficients!(L::AutoDiffVJP{AD}, u, p, t) where{AD <: AutoFiniteDiff}
    update_coefficients!(L.f, u, p, t)
    copy!(L.u, u)
    L
end

# Interpret the call as df/du' * v
function (L::AutoDiffVJP{AD})(v, p, t) where{AD <: AutoFiniteDiff}
    num_vecjac(L.f, L.u, v)
end

function (L::AutoDiffVJP{AD})(dv, v, p, t) where{AD <: AutoFiniteDiff}
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
