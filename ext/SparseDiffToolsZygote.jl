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

function update_coefficients(L::AutoDiffVJP{AD}, u, p, t; VJP_input = nothing,
                            ) where{AD <: AutoZygote}

    if !isnothing(VJP_input)
        @set! L.u = VJP_input
    end

    @set! L.f = update_coefficients(L.f, L.u, p, t)
    @set! L.pullback = Zygote.pullback(L.f, L.u)
end

function update_coefficients!(L::AutoDiffVJP{AD}, u, p, t; VJP_input = nothing,
                             ) where{AD <: AutoZygote}

    if !isnothing(VJP_input)
        copy!(L.u, VJP_input)
    end

    update_coefficients!(L.f, L.u, p, t)
    L.pullback = Zygote.pullback(L.f, L.u)

    L
end

# Interpret the call as df/du' * v
function (L::AutoDiffVJP{AD})(v, p, t; VJP_input = nothing) where{AD <: AutoZygote}
    # ignore VJP_input as pullback was computed in update_coefficients(...)

    y, back = L.pullback
    V = reshape(v, size(y))

    back(V)[1] |> vec
end

# prefer non in-place method
function (L::AutoDiffVJP{AD, IIP, true})(dv, v, p, t; VJP_input = nothing) where {AD <: AutoZygote, IIP}
    # ignore VJP_input as pullback was computed in update_coefficients!(...)

    _dv = L(v, p, t)
    copy!(dv, _dv)
end

function (L::AutoDiffVJP{AD, true, false})(dv, v, p, t; VJP_input = nothing) where {AD <: AutoZygote}
    @error("Zygote requires an out of place method with signature f(u).")
end

end # module
