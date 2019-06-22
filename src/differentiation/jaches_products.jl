struct DeivVecTag end

# J(f(x))*v
function auto_jacvec!(du, f, x, v,
                      cache1 = ForwardDiff.Dual{DeivVecTag}.(x, v),
                      cache2 = ForwardDiff.Dual{DeivVecTag}.(x, v))
    cache1 .= Dual{DeivVecTag}.(x, v)
    f(cache2,cache1)
    du .= partials.(cache2, 1)
end
function auto_jacvec(f, x, v)
    partials.(f(Dual{DeivVecTag}.(x, v)), 1)
end

function num_jacvec!(du,f,x,v,cache1 = similar(v),
                     cache2 = similar(v);
                     compute_f0 = true)
    compute_f0 && (f(cache1,x))
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    @. x += ϵ*v
    f(cache2,x)
    @. x -= ϵ*v
    @. du = (cache2 - cache1)/ϵ
end

function num_jacvec(f,x,v,f0=nothing)
    f0 === nothing ? _f0 = f(x) : _f0 = f0
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(minimum(x)))
    (f(x.+ϵ.*v) .- f(x))./ϵ
end

function num_hesvec!(du,f,x,v,
                     cache1 = similar(v),
                     cache2 = similar(v),
                     cache3 = similar(v),
                     cache4 = similar(v))
    cache = DiffEqDiffTools.GradientCache(cache1,cache2,Val{:central})
    g = (dx,x) -> DiffEqDiffTools.finite_difference_gradient!(dx,f,x,cache)
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    @. x += ϵ*v
    g(cache3,x)
    @. x -= 2ϵ*v
    g(cache4,x)
    @. du = (cache3 - cache4)/(2ϵ)
end

function num_hesvec(f,x,v)
    g = (x) -> DiffEqDiffTools.finite_difference_gradient(f,x)
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    x += ϵ*v
    gxp = g(x)
    x -= 2ϵ*v
    gxm = g(x)
    (gxp - gxm)/(2ϵ)
end


### Operator Forms

mutable struct JacVec{F,T1,T2,uType}
    f::F
    cache1::T1
    cache2::T2
    u::uType
    autodiff::Bool
end

function JacVec(f,u::AbstractArray;autodiff=true)
    if autodiff
        cache1 = ForwardDiff.Dual{DeivVecTag}.(u, u)
        cache2 = ForwardDiff.Dual{DeivVecTag}.(u, u)
    else
        cache1 = similar(u)
        cache2 = similar(u)
    end
    JacVec(f,cache1,cache2,u,autodiff)
end

Base.size(L::JacVec) = (length(L.cache1),length(L.cache1))
Base.size(L::JacVec,i::Int) = length(L.cache1)
Base.:*(L::JacVec,x::AbstractVector) = L.autodiff ? auto_jacvec(_u->L.f(_u),L.u,x) : num_jacvec(_u->L.f(_u),L.u,x)

function LinearAlgebra.mul!(du::AbstractVector,L::JacVec,v::AbstractVector)
    if L.autodiff
        auto_jacvec!(du,(_du,_u)->L.f(_du,_u),L.u,v,L.cache1,L.cache2)
    else
        num_jacvec!(du,(_du,_u)->L.f(_du,_u),L.u,v,L.cache1,L.cache2)
    end
end

mutable struct HesVec{F,T1,T2,uType}
    f::F
    cache1::T1
    cache2::T1
    cache3::T2
    cache4::T2
    u::uType
    autodiff::Bool
end

function HesVec(f;autodiff=false)
    HesVec(f,nothing,nothing,nothing,autodiff)
end

function HesVec(f,u::AbstractArray;autodiff=false)
    if autodiff
        cache1 = ForwardDiff.Dual{DeivVecTag}.(u, u)
        cache2 = ForwardDiff.Dual{DeivVecTag}.(u, u)
    else
        cache1 = similar(u)
        cache2 = similar(u)
        cache3 = similar(u)
        cache4 = similar(u)
    end
    HesVec(f,cache1,cache2,cache3,cache4,u,autodiff)
end

Base.size(L::HesVec) = (length(L.cache1),length(L.cache1))
Base.size(L::HesVec,i::Int) = length(L.cache1)
Base.:*(L::HesVec,x::AbstractVector) = L.autodiff ? @error("Autodiff HesVec is not implemented yet") : num_hesvec(_u->L.f(_u),L.u,x)

function LinearAlgebra.mul!(du::AbstractVector,L::HesVec,v::AbstractVector)
    if L.cache1 === nothing
        if L.autodiff
            @error "Autodiff HesVec is not implemented yet"
        else
            num_hesvec!(du,L.f,L.u,v;compute_f0=false)
        end
    else
        if L.autodiff
            @error "Autodiff HesVec is not implemented yet"
        else
            num_hesvec!(du,L.f,L.u,v,L.cache1,L.cache2,L.cache3,L.cache4)
        end
    end
end
