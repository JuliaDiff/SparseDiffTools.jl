function auto_vecjac!(du, f, x, v, cache1 = nothing, cache2 = nothing)
    !hasmethod(f, (typeof(x),)) &&
        error("For inplace function use autodiff = false")
    du .= auto_vecjac(f, x, v)
end

function auto_vecjac(f, x, v)
    vv, back = Zygote.pullback(f, x)
    return vec(back(reshape(v, size(vv)))[1])
end

function num_vecjac!(
    du,
    f,
    x,
    v,
    cache1 = similar(v),
    cache2 = similar(v);
    compute_f0 = true,
)
    if DiffEqBase.numargs(f) != 2
        du .= num_jacvec(f, x, v)
        return du
    end
    compute_f0 && (f(cache1, x))
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    vv = reshape(v, size(x))
    for i = 1:length(x)
        x[i] += ϵ
        f(cache2, x)
        x[i] -= ϵ
        du[i] = (((cache2 .- cache1) ./ ϵ)'*vv)[1]
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
    for i = 1:length(x)
        x[i] += ϵ
        f0 = f(x)
        x[i] -= ϵ
        du[i] = (((f0 .- _f0) ./ ϵ)'*vv)[1]
    end
    return vec(du)
end