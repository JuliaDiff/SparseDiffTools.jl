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

struct RevModeAutoDiffVecProd{ad,iip,oop,F,U,C,V,V!} <: AbstractAutoDiffVecProd
    f::F
    u::U
    cache::C
    vecprod::V
    vecprod!::V!

    function RevModeAutoDiffVecProd(f, u, cache, vecprod, vecprod!;
                                    autodiff = AutoFiniteDiff(),
                                    isinplace = false, outofplace = true)
        @assert isinplace || outofplace

        new{
            typeof(autodiff),
            isinplace,
            outofplace,
            typeof(f),
            typeof(u),
            typeof(cache),
            typeof(vecprod),
            typeof(vecprod!),
           }(
             f, u, cache, vecprod, vecprod!,
            )
    end
end

function update_coefficients(L::RevModeAutoDiffVecProd, u, p, t)
    RevModeAutoDiffVecProd(L.f, u, L.vecprod, L.vecprod!, L.cache)
end

function update_coefficients!(L::RevModeAutoDiffVecProd, u, p, t)
    copy!(L.u, u)
    L
end

# Interpret the call as df/du' * u
function (L::RevModeAutoDiffVecProd)(v, p, t)
    L.vecprod(_u -> L.f(_u, p, t), L.u, v)
end

# prefer non in-place method
function (L::RevModeAutoDiffVecProd{ad,iip,true})(dv, v, p, t) where{ad,iip}
    L.vecprod!(dv, _u -> L.f(_u, p, t), L.u, v, L.cache...)
end

function (L::RevModeAutoDiffVecProd{ad,true,false})(dv, v, p, t) where{ad}
    L.vecprod!(dv, (_du, _u) -> L.f(_du, _u, p, t), L.u, v, L.cache...)
end

function VecJac(f, u::AbstractArray, p = nothing, t = nothing; autodiff = AutoFiniteDiff(),
                kwargs...)

    vecprod, vecprod! = if autodiff isa AutoFiniteDiff
        num_vecjac, num_vecjac!
    elseif autodiff isa AutoZygote
        @assert static_hasmethod(auto_vecjac, typeof((f, u, u))) "To use AutoZygote() AD, first load Zygote with `using Zygote`, or `import Zygote`"

        auto_vecjac, auto_vecjac!
    end

    cache = (similar(u), similar(u),)

    outofplace = static_hasmethod(f, typeof((u, p, t)))
    isinplace  = static_hasmethod(f, typeof((u, u, p, t)))

    if !(isinplace) & !(outofplace)
        error("$f must have signature f(u, p, t), or f(du, u, p, t)")
    end

    L = RevModeAutoDiffVecProd(f, u, cache, vecprod, vecprod!; autodiff = autodiff,
                               isinplace = isinplace, outofplace = outofplace)

    FunctionOperator(L, u, u;
                     isinplace = isinplace, outofplace = outofplace,
                     p = p, t = t, islinear = true,
                     kwargs...
                    )
end
#
