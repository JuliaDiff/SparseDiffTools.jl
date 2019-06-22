struct JacVecTag end

# J(f(x))*v
function auto_jacvec!(du, f, x, v,
                      cache1 = ForwardDiff.Dual{JacVecTag}.(x, v),
                      cache2 = ForwardDiff.Dual{JacVecTag}.(x, v))
    cache1 .= Dual{JacVecTag}.(x, v)
    f(cache2,cache1)
    du .= partials.(cache2, 1)
end
function auto_jacvec(f, x, v)
    partials.(f(Dual{JacVecTag}.(x, v)), 1)
end

function num_jacvec!(du,f,x,v,cache1 = similar(v),
                     cache2 = similar(v);
                     compute_f0 = true)
    compute_f0 && (f(cache1,x))
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(minimum(x)))
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
