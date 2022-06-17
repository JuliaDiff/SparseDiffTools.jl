struct ForwardColorHesCache{THS, THC, TI<:Integer, TD, TGF, TGC, TG}
    sparsity::THS
    colors::THC
    ncolors::TI
    D::TD
    buffer::TD
    grad!::TGF
    grad_config::TGC
    G::TG
    dG::TG
end

function make_hessian_buffers(colorvec, x)
    ncolors = maximum(colorvec)
    D = hcat([float.(i .== colorvec) for i in 1:ncolors]...)
    buffer = similar(D)
    G = similar(x)
    dG = similar(x)
    return (;ncolors, D, buffer, G, dG)
end

function ForwardColorHesCache(f, 
                              x::AbstractVector{<:Number}, 
                              g!,
                              colorvec::AbstractVector{<:Integer}, 
                              sparsity::AbstractMatrix{<:Integer})
    ncolors, D, buffer, G, dG = make_hessian_buffers(colorvec, x)
    grad_config = ForwardDiff.GradientConfig(f, x)
    return ForwardColorHesCache(sparsity, colorvec, ncolors, D, buffer, g!, grad_config, G, dG)
end


function ForwardColorHesCache(f, 
                              x::AbstractVector{<:Number},
                              colorvec::AbstractVector{<:Integer},
                              sparsity::AbstractMatrix{<:Integer})
    g!(G, x, grad_config) = ForwardDiff.gradient!(G, f, x, grad_config)
    return ForwardColorHesCache(f, x, g!, colorvec, sparsity)
end


function forwarddiff_color_hessian!(H::AbstractMatrix{<:Number}, 
                                    f, 
                                    x::AbstractArray{<:Number}, 
                                    hes_cache::ForwardColorHesCache)
    ϵ = sqrt(eps(eltype(x)))
    for j in 1:hes_cache.ncolors
        hes_cache.grad!(hes_cache.G, x, hes_cache.grad_config)
        x .+= ϵ .* @view hes_cache.D[:, j]
        hes_cache.grad!(hes_cache.dG, x, hes_cache.grad_config)
        x .-= ϵ .* @view hes_cache.D[:, j]
        hes_cache.buffer[:, j] .= (hes_cache.dG .- hes_cache.G) ./ ϵ
    end
    ii, jj, vv = findnz(hes_cache.sparsity)
    for (i, j) in zip(ii, jj)
        H[i, j] = hes_cache.buffer[i, hes_cache.colors[j]]
    end
    return H
end

function forwarddiff_color_hessian!(H::AbstractMatrix{<:Number}, 
                                    f, 
                                    x::AbstractArray{<:Number},
                                    colorvec::AbstractVector{<:Integer}, 
                                    sparsity::AbstractMatrix{<:Integer})
    hes_cache = ForwardColorHesCache(f, x, colorvec, sparsity)
    forwarddiff_color_hessian!(H, f, x, hes_cache)
    return H
end

function forwarddiff_color_hessian(f, 
                                   x::AbstractArray{<:Number}, 
                                   hes_cache::ForwardColorHesCache)
    ii, jj, vv = findnz(hes_cache.sparsity)
    H = sparse(ii, jj, zeros(eltype(x), length(vv)))
    forwarddiff_color_hessian!(H, f, x, hes_cache)
    return H
end