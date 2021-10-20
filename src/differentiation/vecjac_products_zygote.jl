function auto_vecjac!(du, f, x, v, cache1 = nothing, cache2 = nothing)
    !hasmethod(f, (typeof(x),)) &&
        error("For inplace function use autodiff = false")
    du .= auto_vecjac(f, x, v)
end

function auto_vecjac(f, x, v)
    vv, back = Zygote.pullback(f, x)
    return vec(back(reshape(v, size(vv)))[1])
end
