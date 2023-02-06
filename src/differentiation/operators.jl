#
abstract type AbstractAutoDiffVecProd end

## ForwardDiff Operators

mutable struct FwdModeAutoDiffVecProd{F,U,C,V,V!} <: AbstractAutoDiffVecProd
    f::F
    u::U
    cache::C
    vecprod::V
    vecprod!::V!
end

function update_coefficients(L::FwdModeAutoDiffVecProd, u, p, t)
    FwdModeAutoDiffVecProd(L.f, u, L.vecprod, L.vecprod!, L.cache)
end

function update_coefficients!(L::FwdModeAutoDiffVecProd, u, p, t)
    L.u .= u
    L
end

function (L::FwdModeAutoDiffVecProd)(v, p, t)
    L.vecprod(L.f, L.u, v)
end

function (L::FwdModeAutoDiffVecProd)(dv, v, p, t)
    L.vecprod!(dv, L.f, L.u, v, L.cache...)
end

function JacVecProd(f, u::AbstractArray, p = nothing, t = nothing; autodiff = true)

    # signature
    # vecprod(_u -> L.f(_u), L.u, v) # *(L, v)
    # vecprod!(dv, (_v, _u) -> L.f(_v, _u), L.u, v, L.cache1, L.cache2) # mul!(dv,L,v)

    function _f(_u)
        f(_u)
    end

    function _f(_v, _u)
        f(_v, _u)
    end

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

    vecprod  = autodiff ? auto_jacvec  : num_jacvec
    vecprod! = autodiff ? auto_jacvec! : num_jacvec!

    L = FwdModeAutoDiffVecProd(_f, u, cache, vecprod, vecprod!)

    FunctionOperator(L, u, u; # should cache1/cache2 be input/output
                     isinplace = true, outofplace = true,
                     p = p, t = t, islinear = true,
                    )
end

function HesVecProd(f, u::AbstractArray, p = nothing, t = nothing; autodiff = true)

    # signature
    # vecprod(L.f, L.u, v) # *(L, v)
    # vecprod!(dv, L.f, L.u, v, L.cache1, L.cache2, L.cache3) # mul!(dv, L, v)

    if autodiff
        cache1 = ForwardDiff.GradientConfig(f, u)
        cache2 = similar(u)
        cache3 = similar(u)
    else
        cache1 = similar(u)
        cache2 = similar(u)
        cache3 = similar(u)
    end

    cache = (cache1, cache2, cache3,)

    vecprod  = autodiff ? numauto_hesvec  : num_hesvec
    vecprod! = autodiff ? numauto_hesvec! : num_hesvec!

    L = FwdModeAutoDiffVecProd(f, u, cache, vecprod, vecprod!)

    FunctionOperator(L, u, u; # should cache1/cache2 be input/output
                     isinplace = true, outofplace = true,
                     p = p, t = t, islinear = true,
                    )
end

function HesVecGradProd(f, u::AbstractArray, p = nothing, t = nothing;
                        autodiff = true)

    # signature
    # vecprod(L.f, L.u, v) # *(L, v)
    # vecprod!(dv, L.f, L.u, v, L.cache1, L.cache2) # mul!(dy, L, v)

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

    vecprod  = autodiff ? auto_hesvecgrad  : num_hesvecgrad
    vecprod! = autodiff ? auto_hesvecgrad! : num_hesvecgrad!

    L = FwdModeAutoDiffVecProd(f, u, cache, vecprod, vecprod!)

    FunctionOperator(L, u, u; # should cache1/cache2 be input/output
                     isinplace = true, outofplace = true,
                     p = p, t = t, islinear = true,
                    )
end

## Reverse Operators

mutable struct RevModeAutoDiffVecProd{ad,iip,oop,F,U,C,V,V!} <: AbstractAutoDiffVecProd
    f::F
    u::U
    cache::C
    vecprod::V
    vecprod!::V!

    function RevModeAutoDiffVecProd(f, u, cache, vecprod, vecprod!; autodiff = false,
                                    isinplace = false, outofplace = true)
        @assert isinplace || outofplace

        new{
            autodiff,
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

function update_coefficients!(L::RevModeAutoDiffVecProd{true}, u, p, t)
    L.u .= u
    L
end

function update_coefficients!(L::RevModeAutoDiffVecProd{false}, u, p, t)
    L.u .= u
    L.f(L.cache1, L.u, L.p, L.t)
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

function VecJacProd(f, u::AbstractArray, p = nothing, t = nothing; autodiff = true,
                    ishermitian = false, opnrom = true)

    cache = (similar(u), similar(u),)

    vecprod  = autodiff ? auto_vecjac  : num_vecjac
    vecprod! = autodiff ? auto_vecjac! : num_vecjac!

    isinplace  = static_hasmethod(f, typeof((u, p, t)))
    outofplace = static_hasmethod(f, typeof((u, u, p, t)))

    if !(iip) & !(oop)
        error("$f must have signature f(u, p, t), or f(du, u, p, t)")
    end

    L = RevModeAutoDiffVecProd(f, u, vecprod, vecprod!, cache; autodiff = autodiff,
                               isinplace = isinplace, outofplace = outofplace)

    FunctionOperator(L, u, u;
                     isinplace = isinplace, outofplace = outofplace,
                     p = p, t = t, islinear = true,
                     ishermititan = ishermitian, opnorm = opnorm,
                    )
end
#
