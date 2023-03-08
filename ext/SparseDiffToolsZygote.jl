module SparseDiffToolsZygote

import Zygote

using SparseDiffTools
using SparseDiffTools: DeviVecTag, FwdModeAutoDiffVecProd

using ForwardDiff
using ForwardDiff: Dual, Tag

using SciMLOperators: FunctionOperator
using SparseDiffTools.Tricks: static_hasmethod

export
       numback_hesvec, numback_hesvec!,
       autoback_hesvec, autoback_hesvec!,
       auto_vecjac, auto_vecjac!,
       ZygoteVecJac, ZygoteHesVec

### Jac, Hes products

function numback_hesvec!(dy, f, x, v, cache1 = similar(v), cache2 = similar(v))
    g = let f = f
        (dx, x) -> dx .= first(Zygote.gradient(f, x))
    end
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    @. x += ϵ * v
    g(cache1, x)
    @. x -= 2ϵ * v
    g(cache2, x)
    @. dy = (cache1 - cache2) / (2ϵ)
end

function numback_hesvec(f, x, v)
    g = x -> first(Zygote.gradient(f, x))
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    x += ϵ * v
    gxp = g(x)
    x -= 2ϵ * v
    gxm = g(x)
    (gxp - gxm) / (2ϵ)
end

function autoback_hesvec!(dy, f, x, v,
                          cache1 = Dual{typeof(ForwardDiff.Tag(DeivVecTag, eltype(x))),
                                        eltype(x), 1
                                        }.(x,
                                           ForwardDiff.Partials.(Tuple.(reshape(v, size(x))))),
                          cache2 = Dual{typeof(ForwardDiff.Tag(DeivVecTag, eltype(x))),
                                        eltype(x), 1
                                        }.(x,
                                           ForwardDiff.Partials.(Tuple.(reshape(v, size(x))))))
    g = let f = f
        (dx, x) -> dx .= first(Zygote.gradient(f, x))
    end
    cache1 .= Dual{typeof(ForwardDiff.Tag(DeivVecTag, eltype(x))), eltype(x), 1
                   }.(x, ForwardDiff.Partials.(Tuple.(reshape(v, size(x)))))
    g(cache2, cache1)
    dy .= partials.(cache2, 1)
end

function autoback_hesvec(f, x, v)
    g = x -> first(Zygote.gradient(f, x))
    y = Dual{typeof(ForwardDiff.Tag(DeivVecTag, eltype(x))), eltype(x), 1
             }.(x, ForwardDiff.Partials.(Tuple.(reshape(v, size(x)))))
    ForwardDiff.partials.(g(y), 1)
end

# Operator Forms

function ZygoteHesVec(f, u::AbstractArray, p = nothing, t = nothing; autodiff = true)

    if autodiff
        cache1 = Dual{
                      typeof(ForwardDiff.Tag(DeivVecTag(),eltype(u))), eltype(u), 1
                     }.(u, ForwardDiff.Partials.(tuple.(u)))
        cache2 = copy(u)
    else
        cache1 = similar(u)
        cache2 = similar(u)
    end

    cache = (cache1, cache2,)

    vecprod  = autodiff ? autoback_hesvec  : numback_hesvec 
    vecprod! = autodiff ? autoback_hesvec! : numback_hesvec!

    outofplace = static_hasmethod(f, typeof((u,)))
    isinplace  = static_hasmethod(f, typeof((u,)))

    if !(isinplace) & !(outofplace)
        error("$f must have signature f(u).")
    end

    L = FwdModeAutoDiffVecProd(f, u, cache, vecprod, vecprod!)

    FunctionOperator(L, u, u;
                     isinplace = isinplace, outofplace = outofplace,
                     p = p, t = t, islinear = true,
                    )
end

## VecJac products

function auto_vecjac!(du, f, x, v, cache1 = nothing, cache2 = nothing)
    !hasmethod(f, (typeof(x),)) && error("For inplace function use autodiff = false")
    du .= reshape(auto_vecjac(f, x, v), size(du))
end

function auto_vecjac(f, x, v)
    vv, back = Zygote.pullback(f, x)
    return vec(back(reshape(v, size(vv)))[1])
end

ZygoteVecJac(args...; autodiff = true, kwargs...) = VecJac(args...; autodiff = autodiff, kwargs...)

end # module
