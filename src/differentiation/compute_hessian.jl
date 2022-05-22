


# ForwardColorJacCache(f,x,_chunksize = nothing;
#                               dx = nothing,
#                               colorvec=1:length(x),
#                               sparsity = nothing)


# function HessianConfig(logdensity, imarginal, ijoint, forwarddiff_sparsity=false)
# if forwarddiff_sparsity
#     println("Detecting Hessian sparsity via ForwardDiff...")
#     H = ForwardDiff.hessian(f, x)
#     Hsparsity = sparse(H)[imarginal, imarginal] .!= 0
# else
#     println("Detecting Hessian sparsity via SparsityDetection...")
#     Hsparsity = hessian_sparsity(f, x)[imarginal, imarginal]
# end
# Hcolors = matrix_colors(Hsparsity)
#

struct ForwardColorHesCache{THS, THC, TI<:Integer, TD, TG}
    Hsparsity::THS
    Hcolors::THC
    ncolors::TI
    D::TD
    Hcomp_buffer::TD
    G::TG
    δG::TG
end

function ForwardColorHesCache(f, x, colorvec=1:length(x), sparsity=nothing)
    D = hcat([float.(i .== colorvec) for i in 1:maximum(colorvec)]...)
    Hcomp_buffer = similar(D)
    G = zero(x)
    δG = zero(x)
    return ForwardColorHesCache(Hsparsity, colorvec, size(Hcolors, 2), D, Hcomp_buffer, G, δG)
end

function sparse_hessian!(f, g!, θ, hessconfig::ForwardColorHesCache, δ=sqrt(eps(Float64)))
    nc = hessconfig.ncolors
    for j in one(nc):nc
        g!(hessconfig.G, θ)
        g!(hessconfig.δG, θ + δ * @view hessconfig.D[:, j])
        hessconfig.Hcomp_buffer[:, j] .= (hessconfig.δG .- hessconfig.G) ./ δ
    end
    ii, jj, vv = findnz(hessconfig.Hsparsity)
    H = sparse(ii, jj, zeros(length(vv)))
    for (i, j) in zip(ii, jj)
        H[i, j] = hessconfig.Hcomp_buffer[i, hessconfig.Hcolors[j]]
    end
    return H
end

function sparse_hessian!(f, θ, hessconfig::ForwardColorHesCache, δ=sqrt(eps(Float64)))
    g!(G, θ) = ForwardDiff.gradient!(G, f, θ)
    return sparse_hessian!(f, g!, θ, hessconfig, δ)
end
