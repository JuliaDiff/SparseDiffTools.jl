struct DeivVecTag end

get_tag(::Array{Dual{T, V, N}}) where {T, V, N} = T
get_tag(::Dual{T, V, N}) where {T, V, N} = T

# J(f(x))*v
function auto_jacvec!(dy,
                      f,
                      x,
                      v,
                      cache1 = Dual{typeof(ForwardDiff.Tag(DeivVecTag(), eltype(x))),
                                    eltype(x), 1
                                    }.(x,
                                       ForwardDiff.Partials.(tuple.(reshape(v, size(x))))),
                      cache2 = similar(cache1))
    cache1 .= Dual{get_tag(cache1), eltype(x), 1
                   }.(x, ForwardDiff.Partials.(tuple.(reshape(v, size(x)))))
    f(cache2, cache1)
    vecdy = _vec(dy)
    vecdy .= partials.(_vec(cache2), 1)
end

_vec(v) = vec(v)
_vec(v::AbstractVector) = v

function auto_jacvec(f, x, v)
    vv = reshape(v, axes(x))
    y = ForwardDiff.Dual{typeof(ForwardDiff.Tag(DeivVecTag(), eltype(x))), eltype(x), 1
                         }.(x, ForwardDiff.Partials.(tuple.(vv)))
    vec(partials.(vec(f(y)), 1))
end

function num_jacvec!(dy,
                     f,
                     x,
                     v,
                     cache1 = similar(v),
                     cache2 = similar(v);
                     compute_f0 = true)
    vv = reshape(v, axes(x))
    compute_f0 && (f(cache1, x))
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    @. x += ϵ * vv
    f(cache2, x)
    @. x -= ϵ * vv
    vecdy = _vec(dy)
    veccache1 = _vec(cache1)
    veccache2 = _vec(cache2)
    @. vecdy = (veccache2 - veccache1) / ϵ
end

function num_jacvec(f, x, v, f0 = nothing)
    vv = reshape(v, axes(x))
    f0 === nothing ? _f0 = f(x) : _f0 = f0
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(minimum(x)))
    vec((f(x .+ ϵ .* vv) .- _f0) ./ ϵ)
end

function num_hesvec!(dy,
                     f,
                     x,
                     v,
                     cache1 = similar(v),
                     cache2 = similar(v),
                     cache3 = similar(v))
    cache = FiniteDiff.GradientCache(v[1], cache1, Val{:central})
    g = let f = f, cache = cache
        (dx, x) -> FiniteDiff.finite_difference_gradient!(dx, f, x, cache)
    end
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    @. x += ϵ * v
    g(cache2, x)
    @. x -= 2ϵ * v
    g(cache3, x)
    @. dy = (cache2 - cache3) / (2ϵ)
end

function num_hesvec(f, x, v)
    g = (x) -> FiniteDiff.finite_difference_gradient(f, x)
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    x += ϵ * v
    gxp = g(x)
    x -= 2ϵ * v
    gxm = g(x)
    (gxp - gxm) / (2ϵ)
end

function numauto_hesvec!(dy,
                         f,
                         x,
                         v,
                         cache = ForwardDiff.GradientConfig(f, v),
                         cache1 = similar(v),
                         cache2 = similar(v))
    g = let f = f, x = x, cache = cache
        g = (dx, x) -> ForwardDiff.gradient!(dx, f, x, cache)
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

function numauto_hesvec(f, x, v)
    g = (x) -> ForwardDiff.gradient(f, x)
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    x += ϵ * v
    gxp = g(x)
    x -= 2ϵ * v
    gxm = g(x)
    (gxp - gxm) / (2ϵ)
end

function autonum_hesvec!(dy,
                         f,
                         x,
                         v,
                         cache1 = Dual{typeof(ForwardDiff.Tag(DeivVecTag(), eltype(x))),
                                       eltype(x), 1
                                       }.(x,
                                          ForwardDiff.Partials.(tuple.(reshape(v, size(x))))),
                         cache2 = Dual{typeof(ForwardDiff.Tag(DeivVecTag(), eltype(x))),
                                       eltype(x), 1
                                       }.(x,
                                          ForwardDiff.Partials.(tuple.(reshape(v, size(x))))))
    cache = FiniteDiff.GradientCache(v[1], cache1, Val{:central})
    g = (dx, x) -> FiniteDiff.finite_difference_gradient!(dx, f, x, cache)
    cache1 .= Dual{get_tag(cache1), eltype(x), 1
                   }.(x, ForwardDiff.Partials.(tuple.(reshape(v, size(x)))))
    g(cache2, cache1)
    dy .= partials.(cache2, 1)
end

function autonum_hesvec(f, x, v)
    g = (x) -> FiniteDiff.finite_difference_gradient(f, x)
    partials.(g(Dual{DeivVecTag}.(x, v)), 1)
end

function num_hesvecgrad!(dy, g, x, v, cache2 = similar(v), cache3 = similar(v))
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    @. x += ϵ * v
    g(cache2, x)
    @. x -= 2ϵ * v
    g(cache3, x)
    @. dy = (cache2 - cache3) / (2ϵ)
end

function num_hesvecgrad(g, x, v)
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    x += ϵ * v
    gxp = g(x)
    x -= 2ϵ * v
    gxm = g(x)
    (gxp - gxm) / (2ϵ)
end

function auto_hesvecgrad!(dy,
                          g,
                          x,
                          v,
                          cache2 = Dual{typeof(ForwardDiff.Tag(DeivVecTag(), eltype(x))),
                                        eltype(x), 1
                                        }.(x,
                                           ForwardDiff.Partials.(tuple.(reshape(v, size(x))))),
                          cache3 = Dual{typeof(ForwardDiff.Tag(DeivVecTag(), eltype(x))),
                                        eltype(x), 1
                                        }.(x,
                                           ForwardDiff.Partials.(tuple.(reshape(v, size(x))))))
    cache2 .= Dual{get_tag(cache2), eltype(x), 1
                   }.(x, ForwardDiff.Partials.(tuple.(reshape(v, size(x)))))
    g(cache3, cache2)
    dy .= partials.(cache3, 1)
end

function auto_hesvecgrad(g, x, v)
    y = Dual{typeof(ForwardDiff.Tag(DeivVecTag(), eltype(x))), eltype(x), 1
             }.(x, ForwardDiff.Partials.(tuple.(reshape(v, size(x)))))
    partials.(g(y), 1)
end

### Operator Forms
#
# match *, mul! definitions
# https://github.com/JuliaDiff/SparseDiffTools.jl/blob/master/src/differentiation/jaches_products.jl
#
# just make it work.
#

abstract type AbstractAutoDiffVecProd end

mutable struct ForwardDiffVecProd{iip,oop,Tu,F,V,V!,C} <: AbstractAutoDiffVecProd
    u::Tu
    f::F
    vecprod::V
    vecprod!::V!
    cache::C

    function ForwardDiffVecProd(u, f, vecprod, vecprod!, cache;
                             isinplace = nothing,
                             outofplace = nothing,
                            )
        new{
            isinplace,
            outofplace,
            typeof(u),
            typeof(f),
            typeof(vecprod),
            typeof(vecprod!),
            typeof(cache)
           }(
             u, f, vecprod, vecprod!, cache,
            )
    end
end

function update_coefficients(A::ForwardDiffVecProd, u, p, t)
    ForwardDiffVecProd(u, A.f, A.vecprod, A.vecprod!, A.cache)
end

function update_coefficients!(A::ForwardDiffVecProd, u, p, t)
    A.u .= u
    A
end

function (L::ForwardDiffVecProd{false})(v, p, t)
    L.vecprod(L.f, L.u, v)
end

function (L::ForwardDiffVecProd{true})(v, p, t)
    L.vecprod(L.f, L.u, v)
end

function (L::ForwardDiffVecProd)(du, v, p, t)
    L.vecprod!(du, L.f, L.u, v, L.cache...)
end

function JacVec(f, u::AbstractArray, p = nothing, t = nothing; autodiff = true)

    # fix function signature here

    vecprod  = autodiff ? auto_jacvec  : num_jacvec
    vecprod! = autodiff ? auto_jacvec! : num_jacvec!

    if autodiff
        cache1 = Dual{
                      typeof(ForwardDiff.Tag(DeivVecTag(),eltype(u))), eltype(u), 1
                     }.(u, ForwardDiff.Partials.(tuple.(u)))

        cache2 = copy(cache1)
    else
        cache1 = similar(u)
        cache2 = similar(u)
    end

    cache = (cache1, cache2,)

    isinplace = static_hasmethod(f, typeof((u, p, t)))
    outofplace = static_hasmethod(f, typeof((u, u, p, t)))

    L = ForwardDiffVecProd(u, f, vecprod, vecprod!, cache;
                        isinplace = isinplace, outofplace = outofplace)


    FunctionOperator(L, u, u;
                     isinplace = isinplace, outofplace = outofplace,
                     p = p, t = t, islinear = true,
                    )
end

function HesVec(f, u::AbstractArray, p = nothing, t = nothing; autodiff = true)

    # fix function signature here

    vecprod  = autodiff ? numauto_hesvec  : num_hesvec
    vecprod! = autodiff ? numauto_hesvec! : num_hesvec!

    _f = FWrapper(f,p,t)
    if autodiff
        cache1 = ForwardDiff.GradientConfig(_f, u)
        cache2 = similar(u)
        cache3 = similar(u)
    else
        cache1 = similar(u)
        cache2 = similar(u)
        cache3 = similar(u)
    end

    cache = (cache1, cache2, cache3,)

    isinplace = static_hasmethod(f, typeof((u, p, t)))
    outofplace = static_hasmethod(f, typeof((u, u, p, t)))

    L = ForwardDiffVecProd(u, f, vecprod, vecprod!, cache;
                        isinplace = isinplace, outofplace = outofplace)


    FunctionOperator(L, u, u;
                     isinplace = isinplace, outofplace = outofplace,
                     p = p, t = t, islinear = true,
                    )
end

function HesVecGrad(g, u::AbstractArray, p = nothing, t = nothing; autodiff = false)

    # fix function signature here

    vecprod  = autodiff ? numauto_hesvecgrad  : num_hesvecgrad
    vecprod! = autodiff ? numauto_hesvecgrad! : num_hesvecgrad!

    if autodiff
        cache1 = Dual{
                      typeof(ForwardDiff.Tag(DeivVecTag(), eltype(u))), eltype(u), 1
                     }.(u, ForwardDiff.Partials.(tuple.(u)))

        cache2 = copy(cache1)
    else
        cache1 = similar(u)
        cache2 = similar(u)
    end

    cache = (cache1, cache2,)

    isinplace = static_hasmethod(f, typeof((u, p, t)))
    outofplace = static_hasmethod(f, typeof((u, u, p, t)))

    L = ForwardDiffVecProd(u, f, vecprod, vecprod!, cache;
                        isinplace = isinplace, outofplace = outofplace)

    FunctionOperator(L, u, u;
                     isinplace = isinplace, outofplace = outofplace,
                     p = p, t = t, islinear = true,
                    )
end
#
