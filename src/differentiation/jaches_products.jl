struct DeivVecTag end

get_tag(::AbstractArray{Dual{T, V, N}}) where {T, V, N} = T
get_tag(::Dual{T, V, N}) where {T, V, N} = T

# J(f(x))*v
function auto_jacvec!(dy, f, x, v,
        cache1 = Dual{typeof(ForwardDiff.Tag(DeivVecTag(), eltype(x))),
            eltype(x), 1
        }.(x, ForwardDiff.Partials.(tuple.(reshape(v, size(x))))),
        cache2 = similar(cache1))
    cache1 .= Dual{
        get_tag(cache1),
        eltype(x),
        1
    }.(x, ForwardDiff.Partials.(tuple.(reshape(v, size(x)))))
    f(cache2, cache1)
    vecdy = _vec(dy)
    vecdy .= partials.(_vec(cache2), 1)
end

_vec(v) = vec(v)
_vec(v::AbstractVector) = v

function auto_jacvec(f, x, v)
    vv = reshape(v, axes(x))
    y = ForwardDiff.Dual{
        typeof(ForwardDiff.Tag(DeivVecTag(), eltype(x))),
        eltype(x),
        1
    }.(x, ForwardDiff.Partials.(tuple.(vv)))
    vec(partials.(vec(f(y)), 1))
end

function num_jacvec!(dy, f, x, v, cache1 = similar(v), cache2 = similar(v),
        cache3 = similar(v); compute_f0 = true)
    vv = reshape(v, axes(x))
    compute_f0 && (f(cache1, x))
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    @. cache3 = x + ϵ * vv
    f(cache2, cache3)
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

function num_hesvec!(dy, f, x, v, cache1 = similar(v), cache2 = similar(v),
        cache3 = similar(v), cache4 = similar(v))
    cache = FiniteDiff.GradientCache(v[1], cache1, Val{:central})
    g = let f = f, cache = cache
        (dx, x) -> FiniteDiff.finite_difference_gradient!(dx, f, x, cache)
    end
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    @. cache4 = x + ϵ * v
    g(cache2, cache4)
    @. cache4 = x - ϵ * v
    g(cache3, cache4)
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

function numauto_hesvec!(dy, f, x, v, cache = ForwardDiff.GradientConfig(f, v),
        cache1 = similar(v), cache2 = similar(v), cache3 = similar(v))
    g = let f = f, x = x, cache = cache
        g = (dx, x) -> ForwardDiff.gradient!(dx, f, x, cache)
    end
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    @. cache3 = x + ϵ * v
    g(cache1, cache3)
    @. cache3 = x - ϵ * v
    g(cache2, cache3)
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

function autonum_hesvec!(dy, f, x, v,
        cache1 = Dual{typeof(ForwardDiff.Tag(DeivVecTag(), eltype(x))),
            eltype(x), 1
        }.(x, ForwardDiff.Partials.(tuple.(reshape(v, size(x))))),
        cache2 = Dual{typeof(ForwardDiff.Tag(DeivVecTag(), eltype(x))),
            eltype(x), 1
        }.(x, ForwardDiff.Partials.(tuple.(reshape(v, size(x))))))
    cache = FiniteDiff.GradientCache(v[1], cache1, Val{:central})
    g = (dx, x) -> FiniteDiff.finite_difference_gradient!(dx, f, x, cache)
    cache1 .= Dual{
        get_tag(cache1),
        eltype(x),
        1
    }.(x, ForwardDiff.Partials.(tuple.(reshape(v, size(x)))))
    g(cache2, cache1)
    dy .= partials.(cache2, 1)
end

function autonum_hesvec(f, x, v)
    g = (x) -> FiniteDiff.finite_difference_gradient(f, x)
    partials.(g(Dual{DeivVecTag}.(x, v)), 1)
end

function num_hesvecgrad!(dy, g, x, v, cache1 = similar(v), cache2 = similar(v),
        cache3 = similar(v))
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    @. cache3 = x + ϵ * v
    g(cache1, cache3)
    @. cache3 = x - ϵ * v
    g(cache2, cache3)
    @. dy = (cache1 - cache2) / (2ϵ)
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

function auto_hesvecgrad!(dy, g, x, v,
        cache2 = Dual{typeof(ForwardDiff.Tag(DeivVecTag(), eltype(x))),
            eltype(x), 1
        }.(x, ForwardDiff.Partials.(tuple.(reshape(v, size(x))))),
        cache3 = Dual{typeof(ForwardDiff.Tag(DeivVecTag(), eltype(x))),
            eltype(x), 1
        }.(x, ForwardDiff.Partials.(tuple.(reshape(v, size(x))))))
    cache2 .= Dual{
        get_tag(cache2),
        eltype(x),
        1
    }.(x, ForwardDiff.Partials.(tuple.(reshape(v, size(x)))))
    g(cache3, cache2)
    dy .= partials.(cache3, 1)
end

function auto_hesvecgrad(g, x, v)
    y = Dual{
        typeof(ForwardDiff.Tag(DeivVecTag(), eltype(x))),
        eltype(x),
        1
    }.(x, ForwardDiff.Partials.(tuple.(reshape(v, size(x)))))
    partials.(g(y), 1)
end

### Operator Forms

struct FwdModeAutoDiffVecProd{F, U, C, V, V!} <: AbstractAutoDiffVecProd
    f::F
    u::U
    cache::C
    vecprod::V
    vecprod!::V!
end

function update_coefficients(L::FwdModeAutoDiffVecProd, u, p, t)
    @set! L.f = update_coefficients(L.f, u, p, t)
    @set! L.u = u
end

function update_coefficients!(L::FwdModeAutoDiffVecProd, u, p, t)
    update_coefficients!(L.f, u, p, t)
    copy!(L.u, u)
    L
end

function (L::FwdModeAutoDiffVecProd)(v, p, t)
    L.vecprod(L.f, L.u, v)
end

function (L::FwdModeAutoDiffVecProd)(dv, v, p, t)
    L.vecprod!(dv, L.f, L.u, v, L.cache...)
end

function Base.resize!(L::FwdModeAutoDiffVecProd, n::Integer)
    hasmethod(resize!, typeof((L.f, n))) && resize!(L.f, n)
    resize!(L.u, n)

    for v in L.cache
        resize!(v, n)
    end
end

"""
    JacVec(f, u, [p, t]; fu = nothing, autodiff = AutoForwardDiff(), tag = DeivVecTag(),
        kwargs...)

Returns SciMLOperators.FunctionOperator which computes jacobian-vector product `df/du * v`.

!!! note

    For non-square jacobians with inplace `f`, `fu` must be specified, else `JacVec` assumes
    a square jacobian.

```julia
L = JacVec(f, u)

L * v         # = df/du * v
mul!(w, L, v) # = df/du * v

L(v, p, t)    # = df/dw * v
L(x, v, p, t) # = df/dw * v
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
function JacVec(f, u::AbstractArray, p = nothing, t = nothing; fu = nothing,
        autodiff = AutoForwardDiff(), tag = DeivVecTag(),
        use_deprecated_ordering::Val = Val(true), kwargs...)
    ff = JacFunctionWrapper(f, fu, u, p, t; use_deprecated_ordering)
    fu === nothing && (fu = __internal_oop(ff) ? ff(u) : u)

    cache, vecprod, vecprod! = if autodiff isa AutoFiniteDiff
        cache1 = similar(fu)
        cache2 = similar(u)
        cache3 = similar(u)

        (cache1, cache2, cache3), num_jacvec, num_jacvec!
    elseif autodiff isa AutoForwardDiff
        cache1 = Dual{
            typeof(ForwardDiff.Tag(tag, eltype(u))), eltype(u), 1
        }.(u, ForwardDiff.Partials.(tuple.(u)))
        cache2 = Dual{
            typeof(ForwardDiff.Tag(tag, eltype(fu))), eltype(fu), 1
        }.(fu, ForwardDiff.Partials.(tuple.(fu)))

        (cache1, cache2), auto_jacvec, auto_jacvec!
    else
        error("Set autodiff to either AutoForwardDiff(), or AutoFiniteDiff()")
    end

    op = FwdModeAutoDiffVecProd(ff, u, cache, vecprod, vecprod!)

    return FunctionOperator(op, u, fu; isinplace = Val(true), outofplace = Val(true), p, t,
        islinear = true, kwargs...)
end

function HesVec(f, u::AbstractArray, p = nothing, t = nothing;
        autodiff = AutoForwardDiff(), tag = DeivVecTag(), kwargs...)
    cache, vecprod, vecprod! = if autodiff isa AutoFiniteDiff
        cache1 = similar(u)
        cache2 = similar(u)
        cache3 = similar(u)

        (cache1, cache2, cache3), num_hesvec, num_hesvec!
    elseif autodiff isa AutoForwardDiff
        cache1 = ForwardDiff.GradientConfig(f, u)
        cache2 = similar(u)
        cache3 = similar(u)

        (cache1, cache2, cache3), numauto_hesvec, numauto_hesvec!
    elseif autodiff isa AutoZygote
        @assert hasmethod(autoback_hesvec, typeof((f, u, u))) "To use AutoZygote() AD, first load Zygote with `using Zygote`, or `import Zygote`"

        cache1 = Dual{
            typeof(ForwardDiff.Tag(tag, eltype(u))), eltype(u), 1
        }.(u, ForwardDiff.Partials.(tuple.(u)))
        cache2 = copy(cache1)

        (cache1, cache2), autoback_hesvec, autoback_hesvec!
    else
        error("Set autodiff to either AutoForwardDiff(), AutoZygote(), or AutoFiniteDiff()")
    end

    outofplace = hasmethod(f, typeof((u,)))
    isinplace = hasmethod(f, typeof((u,)))

    if !(isinplace) & !(outofplace)
        error("$f must have signature f(u).")
    end

    L = FwdModeAutoDiffVecProd(f, u, cache, vecprod, vecprod!)

    return FunctionOperator(L, u, u; isinplace, outofplace, p, t, islinear = true,
        kwargs...)
end

function HesVecGrad(f, u::AbstractArray, p = nothing, t = nothing;
        autodiff = AutoForwardDiff(), tag = DeivVecTag(), kwargs...)
    cache, vecprod, vecprod! = if autodiff isa AutoFiniteDiff
        cache1 = similar(u)
        cache2 = similar(u)

        (cache1, cache2), num_hesvecgrad, num_hesvecgrad!
    elseif autodiff isa AutoForwardDiff
        cache1 = Dual{
            typeof(ForwardDiff.Tag(tag, eltype(u))), eltype(u), 1
        }.(u, ForwardDiff.Partials.(tuple.(u)))
        cache2 = copy(cache1)

        (cache1, cache2), auto_hesvecgrad, auto_hesvecgrad!
    else
        error("Set autodiff to either AutoForwardDiff(), or AutoFiniteDiff()")
    end

    outofplace = hasmethod(f, typeof((u,)))
    isinplace = hasmethod(f, typeof((u, u)))

    if !(isinplace) & !(outofplace)
        error("$f must have signature f(u), or f(du, u).")
    end

    L = FwdModeAutoDiffVecProd(f, u, cache, vecprod, vecprod!)

    return FunctionOperator(L, u, u; isinplace, outofplace, p, t, islinear = true,
        kwargs...)
end
