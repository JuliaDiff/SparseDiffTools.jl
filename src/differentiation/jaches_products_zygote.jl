function numback_hesvec!(du, f, x, v, cache1 = similar(v), cache2 = similar(v))
    g = let f=f
        (dx, x) -> dx .= first(Zygote.gradient(f,x))
    end
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    @. x += ϵ*v
    g(cache1,x)
    @. x -= 2ϵ*v
    g(cache2,x)
    @. du = (cache1 - cache2)/(2ϵ)
end

function numback_hesvec(f, x, v)
    g = x -> first(Zygote.gradient(f,x))
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    x += ϵ*v
    gxp = g(x)
    x -= 2ϵ*v
    gxm = g(x)
    (gxp - gxm)/(2ϵ)
end

function autoback_hesvec!(du, f, x, v, cache2 = ForwardDiff.Dual{Nothing}.(x, v),
                          cache3 = ForwardDiff.Dual{Nothing}.(x, v))
    g = let f=f
        (dx, x) -> dx .= first(Zygote.gradient(f,x))
    end
    cache2 .= Dual{Nothing}.(x, v)
    g(cache3,cache2)
    du .= partials.(cache3, 1)
end

function autoback_hesvec(f, x, v)
    g = x -> first(Zygote.gradient(f,x))
    ForwardDiff.partials.(g(ForwardDiff.Dual{Nothing}.(x, v)), 1)
end
