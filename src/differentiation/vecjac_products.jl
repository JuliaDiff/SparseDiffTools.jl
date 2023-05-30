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

struct RevModeAutoDiffVecProd{ad, iip, oop, F, U, C, V, V!} <: AbstractAutoDiffVecProd
    f::F
    u::U
    cache::C
    vecprod::V
    vecprod!::V!
    autodiff::ad

    function RevModeAutoDiffVecProd(f, u, cache, vecprod, vecprod!, autodiff)

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
            typeof(vecprod),
            typeof(vecprod!)
           }(
             f, u, cache, vecprod, vecprod!, autodiff,
            )
    end
end

function get_iip_oop(::RevModeAutoDiffVecProd{ad, iip, oop}) where{ad, iip, oop}
    iip, oop
end

function update_coefficients(L::RevModeAutoDiffVecProd, u, p, t)
    @set! L.f = update_coefficients(L.f, u, p, t)
    @set! L.u = u
end

function update_coefficients!(L::RevModeAutoDiffVecProd, u, p, t)
    update_coefficients!(L.f, u, p, t)
    copy!(L.u, u)
    L
end

# Interpret the call as df/du' * u
function (L::RevModeAutoDiffVecProd)(v, p, t)
    L.vecprod(L.f, L.u, v)
end

# prefer non in-place method
function (L::RevModeAutoDiffVecProd{ad, iip, true})(dv, v, p, t) where {ad, iip}
    L.vecprod!(dv, L.f, L.u, v, L.cache...)
end

function (L::RevModeAutoDiffVecProd{ad, true, false})(dv, v, p, t) where {ad}
    L.vecprod!(dv, L.f, L.u, v, L.cache...)
end

function Base.resize!(L::RevModeAutoDiffVecProd, n::Integer)

    static_hasmethod(resize!, typeof((L.f, n))) && resize!(L.f, n)
    resize!(L.u, n)

    for v in L.cache
        resize!(v, n)
    end
end

"""
    VecJac(f, u, [p, t]; autodiff = AutoFiniteDiff())

Returns FunctionOperator that computes
"""
function VecJac(f, u::AbstractArray, p = nothing, t = nothing;
                autodiff = AutoFiniteDiff(), kwargs...)

    vecprod, vecprod!, cache = if autodiff isa AutoFiniteDiff
        num_vecjac, num_vecjac!, (similar(u), similar(u))
    elseif autodiff isa AutoZygote
        @assert static_hasmethod(auto_vecjac, typeof((f, u, u))) "To use AutoZygote() AD, first load Zygote with `using Zygote`, or `import Zygote`"

        auto_vecjac, auto_vecjac!, ()
    end

    L = RevModeAutoDiffVecProd(f, u, cache, vecprod, vecprod!, autodiff)

    iip, oop = get_iip_oop(L)

    FunctionOperator(L, u, u; isinplace = iip, outofplace = oop,
                     p = p, t = t, islinear = true, kwargs...)
end


function FixedVecJac(f, u::AbstractArray, p = nothing, t = nothing;
                     autodiff = AutoFiniteDiff(), kwargs...)
    _fixedvecjac(f, u, p, t, autodiff, kwargs)
end

function _fixedvecjac(f, u, p, t, ad::AutoFiniteDiff, kwargs)
end
#
