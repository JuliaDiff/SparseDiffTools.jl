## Hessian tests
using SparseDiffTools
using Symbolics
using ForwardDiff
using LinearAlgebra, SparseArrays

function fscalar(x)
    return -dot(x, x) + 2 * x[2] * x[3]^2
end

x = randn(5)
sparsity = Symbolics.hessian_sparsity(fscalar, x)
colors = matrix_colors(tril(sparsity))
ncolors = maximum(colors)
D = hcat([float.(i .== colors) for i = 1:ncolors]...)
buffer = similar(D)
G1 = zero(x)
G2 = zero(x)

buffers_tup = SparseDiffTools.make_hessian_buffers(colors, x)
@test buffers_tup.ncolors == ncolors
@test buffers_tup.D == D
@test size(buffers_tup.buffer) == size(buffer)
@test eltype(buffers_tup.buffer) == eltype(buffer)
@test typeof(buffers_tup.buffer) == typeof(buffer)
@test size(buffers_tup.G1) == size(G1)
@test eltype(buffers_tup.G1) == eltype(G1)
@test size(buffers_tup.G2) == size(G2)
@test eltype(buffers_tup.G2) == eltype(G2)


gconfig = ForwardDiff.GradientConfig(fscalar, x)
g(x) = ForwardDiff.gradient(fscalar, x)           # allocating
g!(G, x, gconfig) = ForwardDiff.gradient!(G, fscalar, x, gconfig)   # non-allocating

hescache1 = ForwardColorHesCache(sparsity, colors, ncolors, D, buffer, g!, gconfig, G1, G2)
hescache2 = ForwardColorHesCache(fscalar, x, colors, sparsity, g!)
hescache3 = ForwardColorHesCache(fscalar, x, colors, sparsity)
# custom gradient function
hescache4 = ForwardColorHesCache(
    fscalar,
    x,
    colors,
    sparsity,
    (G, x) -> ForwardDiff.gradient!(G, fscalar, x),
)
hescache5 = ForwardColorHesCache(fscalar, x)
# custom gradient has to have 2 or 3 arguments...
@test_throws ArgumentError ForwardColorHesCache(fscalar, x, colors, sparsity, (a) -> 1.0)
@test_throws ArgumentError ForwardColorHesCache(
    fscalar,
    x,
    colors,
    sparsity,
    (a, b, c, d) -> 1.0,
)
# ...and needs to accept (Vector, Vector, ForwardDiff.GradientConfig)
@test_throws ArgumentError ForwardColorHesCache(
    fscalar,
    x,
    colors,
    sparsity,
    (a::Int, b::Int) -> 1.0,
)
@test_throws ArgumentError ForwardColorHesCache(
    fscalar,
    x,
    colors,
    sparsity,
    (a::Int, b::Int, c::Int) -> 1.0,
)

for name in [:sparsity, :colors, :ncolors, :D]
    @eval @test hescache1.$name == hescache2.$name
    @eval @test hescache1.$name == hescache3.$name
    @eval @test hescache1.$name == hescache4.$name
    # hescache5 is the default dense version, so only first axis will match
    @eval @test size(hescache1.$name, 1) == size(hescache5.$name, 1)
end
for name in [:buffer, :G1, :G2]
    @eval @test size(hescache1.$name) == size(hescache2.$name)
    @eval @test size(hescache1.$name) == size(hescache3.$name)
    @eval @test size(hescache1.$name) == size(hescache4.$name)
    # hescache5 is the default dense version, so only first axis will match
    @eval @test size(hescache1.$name, 1) == size(hescache5.$name, 1)

    @eval @test eltype(hescache1.$name) == eltype(hescache2.$name)
    @eval @test eltype(hescache1.$name) == eltype(hescache3.$name)
    @eval @test eltype(hescache1.$name) == eltype(hescache4.$name)
    @eval @test eltype(hescache1.$name) == eltype(hescache5.$name)
end

Hforward = ForwardDiff.hessian(fscalar, x)
for (i, hescache) in enumerate([hescache1, hescache2, hescache3, hescache4, hescache5])
    H = numauto_color_hessian(fscalar, x, colors, sparsity)
    H1 = numauto_color_hessian(fscalar, x, hescache)
    H2 = numauto_color_hessian(fscalar, x)
    @test all(isapprox.(Hforward, H, rtol = 1e-6))
    @test all(isapprox.(H, H1, rtol = 1e-6))
    @test all(isapprox.(H2, H1, rtol = 1e-6))

    H1 = similar(H)
    numauto_color_hessian!(H1, fscalar, x, collect(hescache.colors), hescache.sparsity)
    @test all(isapprox.(H1, H))

    numauto_color_hessian!(H2, fscalar, x)
    @test all(isapprox.(H2, H))

    numauto_color_hessian!(H1, fscalar, x, hescache)
    @test all(isapprox.(H1, H))

    numauto_color_hessian!(H1, fscalar, x, hescache, safe = false)
    @test all(isapprox.(H1, H))

    # the following tests usually pass, but once in a while don't (it's not a big difference
    # in timing on these small matrices, and sometimes its less than the timing variability).
    # Commenting out for now to avoid rare stochastic test failures.
    # # confirm unsafe is faster
    # t_safe = minimum(@elapsed(numauto_color_hessian!(H1, fscalar, x, hescache, safe=true))
    #     for _ in 1:100)
    # t_unsafe = minimum(@elapsed(numauto_color_hessian!(H1, fscalar, x, hescache, safe=false))
    #     for _ in 1:100)
    # @test t_unsafe <= t_safe
end


hescache1 = ForwardAutoColorHesCache(fscalar, x, colors, sparsity)
hescache2 = ForwardAutoColorHesCache(fscalar, x)


for (i, hescache) in enumerate([hescache1, hescache2])

    H = SparseDiffTools.autoauto_color_hessian(fscalar, x, colors, sparsity)
    H1 = SparseDiffTools.autoauto_color_hessian(fscalar, x, hescache)
    H2 = SparseDiffTools.autoauto_color_hessian(fscalar, x)
    @test all(isapprox.(Hforward, H, rtol = 1e-6))
    @test all(isapprox.(H, H1, rtol = 1e-6))
    @test all(isapprox.(H2, H1, rtol = 1e-6))

    H1 = similar(H)

    SparseDiffTools.autoauto_color_hessian!(
        H1,
        fscalar,
        x,
        collect(hescache.colorvec),
        hescache.sparsity,
    )
    @test all(isapprox.(H1, H))

    SparseDiffTools.autoauto_color_hessian!(H2, fscalar, x)
    @test all(isapprox.(H2, H))

    SparseDiffTools.autoauto_color_hessian!(H1, fscalar, x, hescache)
    @test all(isapprox.(H1, H))

end
