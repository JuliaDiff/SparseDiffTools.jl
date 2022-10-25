struct ForwardColorHesCache{THS,THC,TI<:Integer,TD,TGF,TGC,TG}
    sparsity::THS
    colors::THC
    ncolors::TI
    D::TD
    buffer::TD
    grad!::TGF
    grad_config::TGC
    G1::TG
    G2::TG
end

function make_hessian_buffers(colorvec, x)
    ncolors = maximum(colorvec)
    D = hcat([float.(i .== colorvec) for i = 1:ncolors]...)
    buffer = similar(D)
    G1 = similar(x)
    G2 = similar(x)
    return (; ncolors, D, buffer, G1, G2)
end

function ForwardColorHesCache(
    f,
    x::AbstractVector{<:Number},
    colorvec::AbstractVector{<:Integer} = eachindex(x),
    sparsity::Union{AbstractMatrix,Nothing} = nothing,
    g! = (G, x, grad_config) -> ForwardDiff.gradient!(G, f, x, grad_config),
)
    ncolors, D, buffer, G, G2 = make_hessian_buffers(colorvec, x)
    grad_config = ForwardDiff.GradientConfig(f, x)

    # If user supplied their own gradient function, make sure it has the right
    # signature (i.e. g!(G, x) or g!(G, x, grad_config::ForwardDiff.GradientConfig))
    if !hasmethod(g!, (typeof(G), typeof(G), typeof(grad_config)))
        if !hasmethod(g!, (typeof(G), typeof(G)))
            throw(
                ArgumentError(
                    "Signature of `g!` must be either `g!(G, x)` or `g!(G, x, grad_config::ForwardDiff.GradientConfig)`",
                ),
            )
        end
        # define new method that takes a GradientConfig but doesn't use it
        g1!(G, x, grad_config) = g!(G, x)
    else
        g1! = g!
    end

    if sparsity === nothing
        sparsity = sparse(ones(length(x), length(x)))
    end
    return ForwardColorHesCache(
        sparsity,
        colorvec,
        ncolors,
        D,
        buffer,
        g1!,
        grad_config,
        G,
        G2,
    )
end

function numauto_color_hessian!(
    H::AbstractMatrix{<:Number},
    f,
    x::AbstractArray{<:Number},
    hes_cache::ForwardColorHesCache;
    safe = true,
)
    ϵ = cbrt(eps(eltype(x)))
    for j = 1:hes_cache.ncolors
        x .+= ϵ .* @view hes_cache.D[:, j]
        hes_cache.grad!(hes_cache.G2, x, hes_cache.grad_config)
        x .-= 2ϵ .* @view hes_cache.D[:, j]
        hes_cache.grad!(hes_cache.G1, x, hes_cache.grad_config)
        hes_cache.buffer[:, j] .= (hes_cache.G2 .- hes_cache.G1) ./ 2ϵ
        x .+= ϵ .* @view hes_cache.D[:, j] #reset to original value
    end
    ii, jj, vv = findnz(hes_cache.sparsity)
    if safe
        fill!(H, false)
    end
    for (i, j) in zip(ii, jj)
        H[i, j] = hes_cache.buffer[i, hes_cache.colors[j]]
    end
    return H
end

function numauto_color_hessian!(
    H::AbstractMatrix{<:Number},
    f,
    x::AbstractArray{<:Number},
    colorvec::AbstractVector{<:Integer} = eachindex(x),
    sparsity::Union{AbstractMatrix,Nothing} = nothing,
)
    hes_cache = ForwardColorHesCache(f, x, colorvec, sparsity)
    numauto_color_hessian!(H, f, x, hes_cache)
    return H
end

function numauto_color_hessian(
    f,
    x::AbstractArray{<:Number},
    hes_cache::ForwardColorHesCache,
)
    H = convert.(eltype(x), hes_cache.sparsity)
    numauto_color_hessian!(H, f, x, hes_cache)
    return H
end

function numauto_color_hessian(
    f,
    x::AbstractArray{<:Number},
    colorvec::AbstractVector{<:Integer} = eachindex(x),
    sparsity::Union{AbstractMatrix,Nothing} = nothing,
)
    hes_cache = ForwardColorHesCache(f, x, colorvec, sparsity)
    H = convert.(eltype(x), hes_cache.sparsity)
    numauto_color_hessian!(H, f, x, hes_cache)
    return H
end



## autoauto_color_hessian

mutable struct ForwardAutoColorHesCache{TJC,TG,TS,TC}
    jac_cache::TJC
    grad!::TG
    sparsity::TS
    colorvec::TC
end

struct AutoAutoTag end

function ForwardAutoColorHesCache(
    f,
    x::AbstractVector{V},
    colorvec::AbstractVector{<:Integer} = eachindex(x),
    sparsity::Union{AbstractMatrix,Nothing} = nothing,
    tag::ForwardDiff.Tag = ForwardDiff.Tag(AutoAutoTag(), V),
) where {V}

    if sparsity === nothing
        sparsity = sparse(ones(length(x), length(x)))
    end

    chunksize = ForwardDiff.pickchunksize(maximum(colorvec))
    chunk = ForwardDiff.Chunk(chunksize)

    jacobian_config = ForwardDiff.JacobianConfig(f, x, chunk, tag)
    gradient_config = ForwardDiff.GradientConfig(f, jacobian_config.duals, chunk, tag)

    outer_tag = get_tag(jacobian_config.duals)
    g! = (G, x) -> ForwardDiff.gradient!(G, f, x, gradient_config, Val(false))

    jac_cache = ForwardColorJacCache(g!, x; colorvec, sparsity, tag = outer_tag)

    return ForwardAutoColorHesCache(jac_cache, g!, sparsity, colorvec)
end

function autoauto_color_hessian!(
    H::AbstractMatrix{<:Number},
    f,
    x::AbstractArray{<:Number},
    hes_cache::ForwardAutoColorHesCache,
)

    forwarddiff_color_jacobian!(H, hes_cache.grad!, x, hes_cache.jac_cache)
end

function autoauto_color_hessian!(
    H::AbstractMatrix{<:Number},
    f,
    x::AbstractArray{<:Number},
    colorvec::AbstractVector{<:Integer} = eachindex(x),
    sparsity::Union{AbstractMatrix,Nothing} = nothing,
)
    hes_cache = ForwardAutoColorHesCache(f, x, colorvec, sparsity)
    autoauto_color_hessian!(H, f, x, hes_cache)
    return H
end

function autoauto_color_hessian(
    f,
    x::AbstractArray{<:Number},
    hes_cache::ForwardAutoColorHesCache,
)
    H = convert.(eltype(x), hes_cache.sparsity)
    autoauto_color_hessian!(H, f, x, hes_cache)
    return H
end

function autoauto_color_hessian(
    f,
    x::AbstractArray{<:Number},
    colorvec::AbstractVector{<:Integer} = eachindex(x),
    sparsity::Union{AbstractMatrix,Nothing} = nothing,
)
    hes_cache = ForwardAutoColorHesCache(f, x, colorvec, sparsity)
    H = convert.(eltype(x), hes_cache.sparsity)
    autoauto_color_hessian!(H, f, x, hes_cache)
    return H
end
