struct ForwardColorHesCache{THS, THC, TI<:Integer, TD, TG, T<:Number}
    sparsity::THS
    colors::THC
    ncolors::TI
    D::TD
    buffer::TD
    G::TG
    dG::TG
    dx::T
end

function make_hessian_buffers(colorvec, x)
    ncolors = maximum(colorvec)
    D = hcat([float.(i .== colorvec) for i in 1:ncolors]...)
    buffer = similar(D)
    G = zero(x)
    dG = zero(x)
    return (;ncolors, D, buffer, G, dG)
end

function ForwardColorHesCache(f, x::AbstractArray{T}, dx=sqrt(eps(T)), colorvec=1:length(x), sparsity=nothing) where {T<:Number}
    ncolors, D, buffer, G, dG = make_hessian_buffers(colorvec, x)
    return ForwardColorHesCache(sparsity, colorvec, ncolors, D, buffer, G, dG, dx)
end


function forwarddiff_color_hessian!(H, 
                                    f, 
                                    g!, 
                                    x, 
                                    hes_cache::ForwardColorHesCache, 
                                    d=sqrt(eps(Float64)))
    for j in 1:hes_cache.ncolors
        g!(hes_cache.G, x)
        g!(hes_cache.dG, x + hes_cache.dx * @view hes_cache.D[:, j])
        hes_cache.buffer[:, j] .= (hes_cache.dG .- hes_cache.G) ./ hes_cache.dx
    end
    ii, jj, vv = findnz(hes_cache.sparsity)
    for (i, j) in zip(ii, jj)
        H[i, j] = hes_cache.buffer[i, hes_cache.colors[j]]
    end
    return H
end

function forwarddiff_color_hessian!(H::AbstractMatrix{<:Number}, 
                           f::F, 
                           x::AbstractArray{<:Number}, 
                           hes_cache::ForwardColorHesCache) where F
    g!(G, x) = ForwardDiff.gradient!(G, f, x)
    
    H = forwarddiff_color_hessian!(H, f, g!, x, hes_cache, hes_cache.dx)
    return H
end

function forwarddiff_color_hessian!(H::AbstractMatrix{<:Number}, 
                                    f::F, 
                                    x::AbstractArray{<:Number}; 
                                    dx=sqrt(eps()), 
                                    colorvec=1:length(x), 
                                    sparsity=nothing) where F
    hes_cache = ForwardColorHesCache(f, x, dx, colorvec, sparsity)
    forwarddiff_color_hessian!(H, f, x, hes_cache)
    return H
end


function forwarddiff_color_hessian(f::F, 
                                   x::AbstractArray{<:Number}, 
                                   hes_cache::ForwardColorHesCache) where F
    ii, jj, vv = findnz(hes_cache.sparsity)
    H = sparse(ii, jj, zeros(eltype(x), length(vv)))
    forwarddiff_color_hessian!(H, f, x, hes_cache)
    return H
end
