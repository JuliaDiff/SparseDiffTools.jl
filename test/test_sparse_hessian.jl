## Hessian tests
using SparsityDetection, SparseDiffTools
using ForwardDiff
using LinearAlgebra, SparseArrays

function fscalar(x)
    return -dot(x, x) + 2 * x[2] * x[3]^2
end

x = randn(5)
sparsity = hessian_sparsity(fscalar, x)
colors = matrix_colors(tril(sparsity))
dx = sqrt(eps())
ncolors = maximum(colors)
D = hcat([float.(i .== colors) for i in 1:ncolors]...)
buffer = similar(D)
G = zero(x)
dG = zero(x)

buffers_tup = SparseDiffTools.make_hessian_buffers(colors, x)
@test buffers_tup.ncolors == ncolors
@test buffers_tup.D == D
@test size(buffers_tup.buffer) == size(buffer)
@test typeof(buffers_tup.buffer) == typeof(buffer)
@test buffers_tup.G == G
@test buffers_tup.dG == dG

hescache1 = ForwardColorHesCache(sparsity, colors, ncolors, D, buffer, G, dG, dx)
hescache2 = ForwardColorHesCache(fscalar, x, dx, colors, sparsity)


Hforward = ForwardDiff.hessian(fscalar, x)
g(x) = ForwardDiff.gradient(fscalar, x)           # allocating
g!(G, x) = ForwardDiff.gradient!(G, fscalar, x)   # non-allocating

for hescache in [hescache1, hescache2]
    H = forwarddiff_color_hessian(fscalar, x, hescache)
    @test all(isapprox.(Hforward, H, rtol=1e-6))

    H1 = similar(H)
    forwarddiff_color_hessian!(H1, fscalar, g!, x, hescache, dx)
    @test all(isapprox.(H1, H))

    forwarddiff_color_hessian!(H1, fscalar, x, hescache)
    @test all(isapprox.(H1, H))
end
