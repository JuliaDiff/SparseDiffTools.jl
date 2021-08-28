struct DeivVecTag end

# J(f(x))*v
function auto_jacvec!(dy, f, x, v,
                      cache1 = ForwardDiff.Dual{typeof(ForwardDiff.Tag(DeivVecTag,x))}.(x, v),
                      cache2 = ForwardDiff.Dual{typeof(ForwardDiff.Tag(DeivVecTag,x))}.(x, v))
    cache1 .= Dual{typeof(ForwardDiff.Tag(DeivVecTag,eltype(x)))}.(x, v)
    f(cache2,cache1)
    dy .= partials.(cache2, 1)
end
function auto_jacvec(f, x, v)
    partials.(f(Dual{typeof(ForwardDiff.Tag(DeivVecTag,eltype(x)))}.(x, v)), 1)
end

function num_jacvec!(dy,f,x,v,cache1 = similar(v),
                     cache2 = similar(v);
                     compute_f0 = true)
    compute_f0 && (f(cache1,x))
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    @. x += ϵ*v
    f(cache2,x)
    @. x -= ϵ*v
    @. dy = (cache2 - cache1)/ϵ
end

function num_jacvec(f,x,v,f0=nothing)
    f0 === nothing ? _f0 = f(x) : _f0 = f0
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(minimum(x)))
    (f(x.+ϵ.*v) .- _f0)./ϵ
end

function num_hesvec!(dy,f,x,v,
                     cache1 = similar(v),
                     cache2 = similar(v),
                     cache3 = similar(v))
    cache = FiniteDiff.GradientCache(v[1],cache1,Val{:central})
    g = let f=f,cache=cache
        (dx,x) -> FiniteDiff.finite_difference_gradient!(dx,f,x,cache)
    end
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    @. x += ϵ*v
    g(cache2,x)
    @. x -= 2ϵ*v
    g(cache3,x)
    @. dy = (cache2 - cache3)/(2ϵ)
end

function num_hesvec(f,x,v)
    g = (x) -> FiniteDiff.finite_difference_gradient(f,x)
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    x += ϵ*v
    gxp = g(x)
    x -= 2ϵ*v
    gxm = g(x)
    (gxp - gxm)/(2ϵ)
end

function numauto_hesvec!(dy,f,x,v,
                     cache = ForwardDiff.GradientConfig(f,v),
                     cache1 = similar(v),
                     cache2 = similar(v))
    g = let f=f,x=x,cache=cache
        g = (dx,x) -> ForwardDiff.gradient!(dx,f,x,cache)
    end
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    @. x += ϵ*v
    g(cache1,x)
    @. x -= 2ϵ*v
    g(cache2,x)
    @. dy = (cache1 - cache2)/(2ϵ)
end

function numauto_hesvec(f,x,v)
    g = (x) -> ForwardDiff.gradient(f,x)
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    x += ϵ*v
    gxp = g(x)
    x -= 2ϵ*v
    gxm = g(x)
    (gxp - gxm)/(2ϵ)
end

function autonum_hesvec!(dy,f,x,v,
                     cache1 = ForwardDiff.Dual{DeivVecTag}.(x, v),
                     cache2 = ForwardDiff.Dual{DeivVecTag}.(x, v))
    cache = FiniteDiff.GradientCache(v[1],cache1,Val{:central})
    g = (dx,x) -> FiniteDiff.finite_difference_gradient!(dx,f,x,cache)
    cache1 .= Dual{DeivVecTag}.(x, v)
    g(cache2,cache1)
    dy .= partials.(cache2, 1)
end

function autonum_hesvec(f,x,v)
    g = (x) -> FiniteDiff.finite_difference_gradient(f,x)
    partials.(g(Dual{DeivVecTag}.(x, v)), 1)
end

function num_hesvecgrad!(dy,g,x,v,
                     cache2 = similar(v),
                     cache3 = similar(v))
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    @. x += ϵ*v
    g(cache2,x)
    @. x -= 2ϵ*v
    g(cache3,x)
    @. dy = (cache2 - cache3)/(2ϵ)
end

function num_hesvecgrad(g,x,v)
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    x += ϵ*v
    gxp = g(x)
    x -= 2ϵ*v
    gxm = g(x)
    (gxp - gxm)/(2ϵ)
end

function auto_hesvecgrad!(dy,g,x,v,
                     cache2 = ForwardDiff.Dual{DeivVecTag}.(x, v),
                     cache3 = ForwardDiff.Dual{DeivVecTag}.(x, v))
    cache2 .= Dual{DeivVecTag}.(x, v)
    g(cache3,cache2)
    dy .= partials.(cache3, 1)
end

function auto_hesvecgrad(g,x,v)
    partials.(g(Dual{DeivVecTag}.(x, v)), 1)
end

### Operator Forms

struct JacVec{F,T1,T2,xType}
    f::F
    cache1::T1
    cache2::T2
    x::xType
    autodiff::Bool
end

function JacVec(f,x::AbstractArray;autodiff=true)
    if autodiff
        cache1 = ForwardDiff.Dual{DeivVecTag}.(x, x)
        cache2 = ForwardDiff.Dual{DeivVecTag}.(x, x)
    else
        cache1 = similar(x)
        cache2 = similar(x)
    end
    JacVec(f,cache1,cache2,x,autodiff)
end

Base.size(L::JacVec) = (length(L.cache1),length(L.cache1))
Base.size(L::JacVec,i::Int) = length(L.cache1)
Base.:*(L::JacVec,v::AbstractVector) = L.autodiff ? auto_jacvec(_x->L.f(_x),L.x,v) : num_jacvec(_x->L.f(_x),L.x,v)

function LinearAlgebra.mul!(dy::AbstractVector,L::JacVec,v::AbstractVector)
    if L.autodiff
        auto_jacvec!(dy,(_y,_x)->L.f(_y,_x),L.x,v,L.cache1,L.cache2)
    else
        num_jacvec!(dy,(_y,_x)->L.f(_y,_x),L.x,v,L.cache1,L.cache2)
    end
end

struct HesVec{F,T1,T2,xType}
    f::F
    cache1::T1
    cache2::T2
    cache3::T2
    x::xType
    autodiff::Bool
end

function HesVec(f,x::AbstractArray;autodiff=true)
    if autodiff
        cache1 = ForwardDiff.GradientConfig(f,x)
        cache2 = similar(x)
        cache3 = similar(x)
    else
        cache1 = similar(x)
        cache2 = similar(x)
        cache3 = similar(x)
    end
    HesVec(f,cache1,cache2,cache3,x,autodiff)
end

Base.size(L::HesVec) = (length(L.cache2),length(L.cache2))
Base.size(L::HesVec,i::Int) = length(L.cache2)
Base.:*(L::HesVec,v::AbstractVector) = L.autodiff ? numauto_hesvec(L.f,L.x,v) : num_hesvec(L.f,L.x,v)

function LinearAlgebra.mul!(dy::AbstractVector,L::HesVec,v::AbstractVector)
    if L.autodiff
        numauto_hesvec!(dy,L.f,L.x,v,L.cache1,L.cache2,L.cache3)
    else
        num_hesvec!(dy,L.f,L.x,v,L.cache1,L.cache2,L.cache3)
    end
end

struct HesVecGrad{G,T1,T2,uType}
    g::G
    cache1::T1
    cache2::T2
    x::uType
    autodiff::Bool
end

function HesVecGrad(g,x::AbstractArray;autodiff=false)
    if autodiff
        cache1 = ForwardDiff.Dual{DeivVecTag}.(x, x)
        cache2 = ForwardDiff.Dual{DeivVecTag}.(x, x)
    else
        cache1 = similar(x)
        cache2 = similar(x)
    end
    HesVecGrad(g,cache1,cache2,x,autodiff)
end

Base.size(L::HesVecGrad) = (length(L.cache2),length(L.cache2))
Base.size(L::HesVecGrad,i::Int) = length(L.cache2)
Base.:*(L::HesVecGrad,v::AbstractVector) = L.autodiff ? auto_hesvecgrad(L.g,L.x,v) : num_hesvecgrad(L.g,L.x,v)

function LinearAlgebra.mul!(dy::AbstractVector,L::HesVecGrad,v::AbstractVector)
    if L.autodiff
        auto_hesvecgrad!(dy,L.g,L.x,v,L.cache1,L.cache2)
    else
        num_hesvecgrad!(dy,L.g,L.x,v,L.cache1,L.cache2)
    end
end
