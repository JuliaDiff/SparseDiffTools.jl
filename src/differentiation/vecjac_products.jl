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

mutable struct VecJac{iip,T,F,T1,T2,uType,pType,tType,O} <: AbstractSciMLOperator{T}
    f::FWrapper{iip,F,pType,tType}
    cache1::T1
    cache2::T2
    u::uType
    autodiff::Bool
    ishermitian::Bool
    opnorm::O

    function VecJac{T}(f,u::AbstractArray,
                       p = nothing,t::Union{Nothing,Number} = nothing;
                       autodiff = true,
                       ishermitian = false,
                       opnorm = true
                      ) where {T}

        cache1 = similar(u)
        cache2 = similar(u)

        new{t,typeof(f),typeof(cache1),typeof(cache2),
            typeof(u),P,tType,typeof(opnorm)}(
            f,cache1,cache2,u,p,t,autodiff,ishermitian,
            opnorm)
    end

    function VecJac(f, u, args...; kwargs...)
        VecJac{eltype(u)}(f, u, args...; kwargs...)
    end
end

LinearAlgebra.opnorm(L::VecJac, p::Real = 2) = L.opnorm
LinearAlgebra.ishermitian(L::VecJac) = L.ishermitian

Base.size(L::VecJac) = (length(L.cache1), length(L.cache1))
Base.size(L::VecJac, i::Int) = length(L.cache1)

function update_coefficients!(L::VecJac, u, p, t)
    L.u = u
    L.p = p
    L.t = t
    !L.autodiff && L.cache1 !== nothing && L.f(L.cache1, L.u, L.p, L.t)
end

# Interpret the call as df/du' * u
function(L::VecJac)(u, p, t::Number)
    update_coefficients!(L, u, p, t)
    L * u
end

function(L::VecJac)(du, u, p, t::Number)
    update_coefficients!(L, u, p, t)
    mul!(du, L, u)
end


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
                        L.cache1,
                        L.cache2,
                    )
                else
                    auto_vecjac!(
                        du,
                        (_du, _u) -> L.f(_du, _u, p, t),
                        L.u,
                        x,
                        L.cache1,
                        L.cache2,
                    )
                end
            else
                if hasmethod(L.f, typeof.((du, L.u, L.p, L.t)))
                    num_vecjac!(
                        du,
                        (_du, _u) -> L.f(_du, _u, p, t),
                        L.u,
                        x,
                        L.cache1,
                        L.cache2;
                        compute_f0 = true,
                    )
                else
                    num_vecjac!(
                        du,
                        _u -> L.f(_u, p, t),
                        L.u,
                        x,
                        L.cache1,
                        L.cache2;
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
