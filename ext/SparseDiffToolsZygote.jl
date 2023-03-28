module SparseDiffToolsZygote

if isdefined(Base, :get_extension)
    import Zygote
    using LinearAlgebra
    using SparseDiffTools: SparseDiffTools, DeivVecTag
    using ForwardDiff: ForwardDiff, Dual, partials
else
    import ..Zygote
    using ..LinearAlgebra
    using ..SparseDiffTools: SparseDiffTools, DeivVecTag
    using ..ForwardDiff: ForwardDiff, Dual, partials
end

### Jac, Hes products

function SparseDiffTools.numback_hesvec!(dy, f, x, v, cache1 = similar(v), cache2 = similar(v))
    g = let f = f
        (dx, x) -> dx .= first(Zygote.gradient(f, x))
    end
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    @. x += ϵ * v
    g(cache1, x)
    @. x -= 2ϵ * v
    g(cache2, x)
    @. dy = (cache1 - cache2) / (2ϵ)
end

function SparseDiffTools.numback_hesvec(f, x, v)
    g = x -> first(Zygote.gradient(f, x))
    T = eltype(x)
    # Should it be min? max? mean?
    ϵ = sqrt(eps(real(T))) * max(one(real(T)), abs(norm(x)))
    x += ϵ * v
    gxp = g(x)
    x -= 2ϵ * v
    gxm = g(x)
    (gxp - gxm) / (2ϵ)
end

function SparseDiffTools.autoback_hesvec!(dy, f, x, v,
                          cache1 = Dual{typeof(ForwardDiff.Tag(DeivVecTag(), eltype(x))),
                                        eltype(x), 1
                                        }.(x,
                                           ForwardDiff.Partials.(Tuple.(reshape(v, size(x))))),
                          cache2 = Dual{typeof(ForwardDiff.Tag(DeivVecTag(), eltype(x))),
                                        eltype(x), 1
                                        }.(x,
                                           ForwardDiff.Partials.(Tuple.(reshape(v, size(x))))))
    g = let f = f
        (dx, x) -> dx .= first(Zygote.gradient(f, x))
    end
    # Reset each dual number in cache1 to primal = dual = 1.
    cache1 .= eltype(cache1).(x, ForwardDiff.Partials.(Tuple.(reshape(v, size(x)))))
    g(cache2, cache1)
    dy .= partials.(cache2, 1)
end

function SparseDiffTools.autoback_hesvec(f, x, v)
    g = x -> first(Zygote.gradient(f, x))
    y = Dual{typeof(ForwardDiff.Tag(DeivVecTag, eltype(x))), eltype(x), 1
             }.(x, ForwardDiff.Partials.(Tuple.(reshape(v, size(x)))))
    ForwardDiff.partials.(g(y), 1)
end

## VecJac products

function SparseDiffTools.auto_vecjac!(du, f, x, v, cache1 = nothing, cache2 = nothing)
    !hasmethod(f, (typeof(x),)) && error("For inplace function use autodiff = false")
    du .= reshape(SparseDiffTools.auto_vecjac(f, x, v), size(du))
end

function SparseDiffTools.auto_vecjac(f, x, v)
    vv, back = Zygote.pullback(f, x)
    return vec(back(reshape(v, size(vv)))[1])
end

end # module
