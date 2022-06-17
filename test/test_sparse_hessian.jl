## Hessian tests
using SparsityDetection, SparseDiffTools
using ForwardDiff
using LinearAlgebra, SparseArrays

function fscalar(x)
    return -dot(x, x) + 2 * x[2] * x[3]^2
end

x = randn(50)
sparsity = hessian_sparsity(fscalar, x)
colors = matrix_colors(tril(sparsity))
ncolors = maximum(colors)
D = hcat([float.(i .== colors) for i in 1:ncolors]...)
buffer = similar(D)
G = zero(x)
dG = zero(x)

buffers_tup = SparseDiffTools.make_hessian_buffers(colors, x)
@test buffers_tup.ncolors == ncolors
@test buffers_tup.D == D
@test size(buffers_tup.buffer) == size(buffer)
@test eltype(buffers_tup.buffer) == eltype(buffer)
@test typeof(buffers_tup.buffer) == typeof(buffer)
@test size(buffers_tup.G) == size(G)
@test eltype(buffers_tup.G) == eltype(G)
@test size(buffers_tup.dG) == size(dG)
@test eltype(buffers_tup.dG) == eltype(dG)


gconfig = ForwardDiff.GradientConfig(fscalar, x)
g(x) = ForwardDiff.gradient(fscalar, x)           # allocating
g!(G, x, gconfig) = ForwardDiff.gradient!(G, fscalar, x, gconfig)   # non-allocating

hescache1 = ForwardColorHesCache(sparsity, colors, ncolors, D, buffer, g!, gconfig, G, dG)
hescache2 = ForwardColorHesCache(fscalar, x, g!, colors, sparsity)
hescache3 = ForwardColorHesCache(fscalar, x, colors, sparsity)

for name in [:sparsity, :colors, :ncolors, :D]
    @eval @test hescache1.$name == hescache2.$name
    @eval @test hescache1.$name == hescache3.$name
end
for name in [:buffer, :G, :dG]
    @eval @test size(hescache1.$name) == size(hescache2.$name)
    @eval @test size(hescache1.$name) == size(hescache2.$name)
    @eval @test eltype(hescache1.$name) == eltype(hescache3.$name)
    @eval @test eltype(hescache1.$name) == eltype(hescache3.$name)
end

Hforward = ForwardDiff.hessian(fscalar, x)
for hescache in [hescache1, hescache2, hescache3]
    H = forwarddiff_color_hessian(fscalar, x, hescache)
    @test all(isapprox.(Hforward, H, rtol=1e-6))

    H1 = similar(H)
    forwarddiff_color_hessian!(H1, fscalar, x, colors, sparsity)
    @test all(isapprox.(H1, H))

    forwarddiff_color_hessian!(H1, fscalar, x, hescache)
    @test all(isapprox.(H1, H))
end


