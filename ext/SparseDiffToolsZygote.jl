module SparseDiffToolsZygote

import Zygote
using ADTypes
using LinearAlgebra
using SparseDiffTools: SparseDiffTools, DeivVecTag, AutoDiffVJP
using ForwardDiff: ForwardDiff, Dual, partials
import SciMLOperators: update_coefficients, update_coefficients!
import Setfield: @set!

### Jac, Hes products

function SparseDiffTools.numback_hesvec!(dy, f, x, v, cache1 = similar(v),
                                         cache2 = similar(v))
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
    @. x += ϵ * v
    @. dy = (cache1 - cache2) / (2ϵ)
end

function SparseDiffTools.numback_hesvec(f, x, v)
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

function SparseDiffTools.autoback_hesvec!(dy, f, x, v,
                                          cache1 = Dual{
                                                        typeof(ForwardDiff.Tag(DeivVecTag(),
                                                                               eltype(x))),
                                                        eltype(x), 1
                                                        }.(x,
                                                           ForwardDiff.Partials.(tuple.(reshape(v,
                                                                                                size(x))))),
                                          cache2 = Dual{
                                                        typeof(ForwardDiff.Tag(DeivVecTag(),
                                                                               eltype(x))),
                                                        eltype(x), 1
                                                        }.(x,
                                                           ForwardDiff.Partials.(tuple.(reshape(v,
                                                                                                size(x))))))
    g = let f = f
        (dx, x) -> dx .= first(Zygote.gradient(f, x))
    end
    # Reset each dual number in cache1 to primal = dual = 1.
    cache1 .= eltype(cache1).(x, ForwardDiff.Partials.(tuple.(reshape(v, size(x)))))
    g(cache2, cache1)
    dy .= partials.(cache2, 1)
end

function SparseDiffTools.autoback_hesvec(f, x, v)
    g = x -> first(Zygote.gradient(f, x))
    y = Dual{typeof(ForwardDiff.Tag(DeivVecTag(), eltype(x))), eltype(x), 1
             }.(x, ForwardDiff.Partials.(tuple.(reshape(v, size(x)))))
    ForwardDiff.partials.(g(y), 1)
end

## VecJac products

# VJP methods
function SparseDiffTools.auto_vecjac!(du, f, x, v)
    !hasmethod(f, (typeof(x),)) && error("For inplace function use autodiff = AutoFiniteDiff()")
    du .= reshape(SparseDiffTools.auto_vecjac(f, x, v), size(du))
end

function SparseDiffTools.auto_vecjac(f, x, v)
    y, back = Zygote.pullback(f, x)
    return vec(back(reshape(v, size(y)))[1])
end

# overload operator interface
function SparseDiffTools._vecjac(f, u, autodiff::AutoZygote)

    cache = ()
    pullback = Zygote.pullback(f, u)

    AutoDiffVJP(f, u, cache, autodiff, pullback)
end

function update_coefficients(L::AutoDiffVJP{AD}, u, p, t) where{AD <: AutoZygote}
    @set! L.f = update_coefficients(L.f, u, p, t)
    @set! L.u = u
    @set! L.pullback = Zygote.pullback(L.f, u)
end

function update_coefficients!(L::AutoDiffVJP{AD}, u, p, t) where{AD <: AutoZygote}
    update_coefficients!(L.f, u, p, t)
    copy!(L.u, u)
    L.pullback = Zygote.pullback(L.f, u)
    L
end

# Interpret the call as df/du' * v
function (L::AutoDiffVJP{AD})(v, p, t) where{AD <: AutoZygote}

    y, back = L.pullback
    V = reshape(v, size(y))

    back(V)[1] |> vec
end

# prefer non in-place method
function (L::AutoDiffVJP{AD, IIP, true})(dv, v, p, t) where {AD <: AutoZygote, IIP}
    _dv = L(v, p, t)
    copy!(dv, _dv)
end

function (L::AutoDiffVJP{AD, true, false})(dv, v, p, t) where {AD <: AutoZygote}
    SparseDiffTools.auto_vecjac!(dv, L.f, L.u, v, L.cache...)
end

end # module
