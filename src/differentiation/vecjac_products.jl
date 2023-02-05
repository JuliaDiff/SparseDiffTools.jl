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

## Operators

function VecJac(f, u::AbstractArray, p = nothing, t = nothing; autodiff = true)

    vecprod  = autodiff ? __  : __
    vecprod! = autodiff ? __! : __!

    cache = (similar(u), similar(u),)

    isinplace = static_hasmethod(f, typeof((u, p, t)))
    outofplace = static_hasmethod(f, typeof((u, u, p, t)))

    L = AutoDiffVecProd(u, f, vecprod, vecprod!, cache; isinplace = isinplace)

    FunctionOperator(L, u, u;
                     isinplace = isinplace, outofplace = outofplace,
                     p = p, t = t, islinear = true,
                     ishermititan = false, opnorm = true,
                    )
end

function update_coefficients!(L::VecJac, u, p, t)
    L.u = u
    !L.autodiff && L.cache1 !== nothing && L.f(L.cache1, L.u, L.p, L.t)
end

# Interpret the call as df/du' * u

function Base.:*(L::VecJac, x::AbstractVector)
    if hasmethod(L.f, typeof.((L.u, L.p, L.t)))
        return vec(
        L.autodiff ? auto_vecjac(_u -> L.f(_u, L.p, L.t), L.u, x) :
        num_vecjac(_u -> L.f(_u, L.p, L.t), L.u, x),
        )
    end
    return mul!(similar(vec(L.u)), L, x)
end

function LinearAlgebra.mul!(du::AbstractVector,L::VecJac,x::AbstractVector)
    du = reshape(du, size(L.u))
    let p = L.p, t = L.t
        if L.cache1 === nothing
            if L.autodiff
                # For autodiff prefer non-inplace function
                if hasmethod(L.f, typeof.((L.u, L.p, L.t)))
                    auto_vecjac!(du, _u -> L.f(_u, p, t), L.u, x)
                else
                    auto_vecjac!(du, (_du, _u) -> L.f(_du, _u, p, t), L.u, x)
                end
            else
                if hasmethod(L.f, typeof.((du, L.u, L.p, L.t)))
                    num_vecjac!(
                        du,
                        (_du, _u) -> L.f(_du, _u, p, t),
                        L.u,
                        x;
                        compute_f0 = true,
                    )
                else
                    num_vecjac!(
                        du,
                        _u -> L.f(_u, p, t),
                        L.u,
                        x;
                        compute_f0 = true,
                    )
                end
            end
        else
            if L.autodiff
                # For autodiff prefer non-inplace function
                if hasmethod(L.f, typeof.((L.u, L.p, L.t)))
                    auto_vecjac!(
                        du,
                        _u -> L.f(_u, p, t),
                        L.u,
                        x,
                        L.cache...
                    )
                else
                    auto_vecjac!(
                        du,
                        (_du, _u) -> L.f(_du, _u, p, t),
                        L.u,
                        x,
                        L.cache...
                    )
                end
            else
                if hasmethod(L.f, typeof.((du, L.u, L.p, L.t)))
                    num_vecjac!(
                        du,
                        (_du, _u) -> L.f(_du, _u, p, t),
                        L.u,
                        x,
                        L.cache...;
                        compute_f0 = true,
                    )
                else
                    num_vecjac!(
                        du,
                        _u -> L.f(_u, p, t),
                        L.u,
                        x,
                        L.cache...;
                        compute_f0 = true,
                    )
                end
            end
        end
    end
    return vec(du)
end

function Base.resize!(J::VecJac, i)
    resize!(J.cache1, i)
    resize!(J.cache2, i)
end
