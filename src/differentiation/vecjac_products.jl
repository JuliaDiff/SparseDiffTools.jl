function _numargs(f)
    typ = Tuple{Any, Val{:analytic}, Vararg}
    typ2 = Tuple{Any, Type{Val{:analytic}}, Vararg} # This one is required for overloaded types
    typ3 = Tuple{Any, Val{:jac}, Vararg}
    typ4 = Tuple{Any, Type{Val{:jac}}, Vararg} # This one is required for overloaded types
    typ5 = Tuple{Any, Val{:tgrad}, Vararg}
    typ6 = Tuple{Any, Type{Val{:tgrad}}, Vararg} # This one is required for overloaded types
    numparam = maximum([(m.sig<:typ || m.sig<:typ2 || m.sig<:typ3 || m.sig<:typ4 || m.sig<:typ5 || m.sig<:typ6) ? 0 : num_types_in_tuple(m.sig) for m in methods(f)])
    return (numparam-1) #-1 in v0.5 since it adds f as the first parameter
end


#Get the number of parameters of a Tuple type, i.e. the number of fields.

function num_types_in_tuple(sig)
  length(sig.parameters)
end

function num_types_in_tuple(sig::UnionAll)
  length(Base.unwrap_unionall(sig).parameters)
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
    if _numargs(f) != 2
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
    println(typeof(v))
    vv = reshape(v, axes(x))
    f0 === nothing ? _f0 = f(x) : _f0 = f0
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    du = similar(x)
    for i = 1:length(x)
        println(typeof(x))
        x[i] += ϵ
        f0 = f(x)
        x[i] -= ϵ
        du[i] = (((f0 .- _f0) ./ ϵ)'*vv)[1]
    end
    return vec(du)
end
