## Sparse Jacobian tests
using SparseDiffTools, Symbolics, ForwardDiff, LinearAlgebra, SparseArrays, Zygote
using Test

@views function fdiff(y, x) # in-place
    y[(begin + 1):(end - 1)] .= x[begin:(end - 2)] .- 2 .* x[(begin + 1):(end - 1)] .+
                                x[(begin + 2):end]
    y[begin] = -2 * x[begin] + x[begin + 1]
    y[end] = x[end - 1] - 2 * x[end]
    return nothing
end

@views function fdiff(x) # out-of-place
    y₂ = x[begin:(end - 2)] .- 2 .* x[(begin + 1):(end - 1)] .+ x[(begin + 2):end]
    y₁ = -2x[1] + x[2]
    y₃ = x[end - 1] - 2x[end]
    return vcat(y₁, y₂, y₃)
end

x = randn(Float32, 100);

J_true = ForwardDiff.jacobian(fdiff, x);

@info "`ForwardDiff.jacobian` time: $(@belapsed(ForwardDiff.jacobian($fdiff, $x)))s"

# SparseDiffTools High-Level API
J_sparsity = Symbolics.jacobian_sparsity(fdiff, similar(x), x);

SPARSITY_DETECTION_ALGS = [JacPrototypeSparsityDetection(jac_prototype = J_sparsity),
    SymbolicsSparsityDetection(), NoSparsityDetection()]

@testset "Sparsity Detection: $(nameof(typeof(sd)))" for sd in SPARSITY_DETECTION_ALGS
    @info "Sparsity Detection: $(nameof(typeof(sd)))"
    @info "Out of Place Function"
    @testset "sparse_jacobian: Out of Place" begin
        for difftype in (AutoSparseZygote(), AutoZygote(), AutoSparseForwardDiff(),
            AutoForwardDiff(), AutoSparseFiniteDiff(), AutoFiniteDiff())
            @testset "Cache & Reuse" begin
                cache = sparse_jacobian_cache(difftype, sd, fdiff, x)
                J = SparseDiffTools.__init_𝒥(cache)

                sparse_jacobian!(J, difftype, cache, fdiff, x)

                @test J ≈ J_true
                @inferred sparse_jacobian!(J, difftype, cache, fdiff, x)

                t₁ = @belapsed sparse_jacobian!($J, $difftype, $cache, $fdiff, $x)
                @info "$(nameof(typeof(difftype)))() `sparse_jacobian!` (only differentiation) time: $(t₁)s"

                J = sparse_jacobian(difftype, cache, fdiff, x)

                @test J ≈ J_true
                # @inferred sparse_jacobian(difftype, cache, fdiff, x)

                t₂ = @belapsed sparse_jacobian($difftype, $cache, $fdiff, $x)
                @info "$(nameof(typeof(difftype)))() `sparse_jacobian` (with matrix allocation) time: $(t₂)s"
            end

            @testset "Single Use" begin
                J = sparse_jacobian(difftype, sd, fdiff, x)

                @test J ≈ J_true
                # @inferred sparse_jacobian(difftype, sd, fdiff, x)

                t₁ = @belapsed sparse_jacobian($difftype, $sd, $fdiff, $x)
                @info "$(nameof(typeof(difftype)))() `sparse_jacobian` (complete) time: $(t₁)s"

                cache = sparse_jacobian_cache(difftype, sd, fdiff, x)
                J = SparseDiffTools.__init_𝒥(cache)

                sparse_jacobian!(J, difftype, sd, fdiff, x)

                @test J ≈ J_true
                @inferred sparse_jacobian!(J, difftype, sd, fdiff, x)

                t₂ = @belapsed sparse_jacobian!($J, $difftype, $sd, $fdiff, $x)
                @info "$(nameof(typeof(difftype)))() `sparse_jacobian!` (with matrix coloring) time: $(t₂)s"
            end
        end
    end

    @info "Inplace Place Function"
    @testset "sparse_jacobian: In place" begin
        for difftype in (AutoSparseForwardDiff(), AutoForwardDiff(), AutoSparseFiniteDiff(),
            AutoFiniteDiff())
            y = similar(x)
            cache = sparse_jacobian_cache(difftype, sd, fdiff, y, x)
            @testset "Cache & Reuse" begin
                J = SparseDiffTools.__init_𝒥(cache)
                sparse_jacobian!(J, difftype, cache, fdiff, y, x)

                @test J ≈ J_true
                @inferred sparse_jacobian!(J, difftype, cache, fdiff, y, x)

                t₁ = @belapsed sparse_jacobian!($J, $difftype, $cache, $fdiff, $y, $x)
                @info "$(nameof(typeof(difftype)))() `sparse_jacobian!` (only differentiation) time: $(t₁)s"

                J = sparse_jacobian(difftype, cache, fdiff, y, x)

                @test J ≈ J_true
                # @inferred sparse_jacobian(difftype, cache, fdiff, y, x)

                t₂ = @belapsed sparse_jacobian($difftype, $cache, $fdiff, $y, $x)
                @info "$(nameof(typeof(difftype)))() `sparse_jacobian` (with jacobian allocation) time: $(t₂)s"
            end

            @testset "Single Use" begin
                J = sparse_jacobian(difftype, sd, fdiff, y, x)

                @test J ≈ J_true
                # @inferred sparse_jacobian(difftype, sd, fdiff, y, x)

                t₁ = @belapsed sparse_jacobian($difftype, $sd, $fdiff, $y, $x)
                @info "$(nameof(typeof(difftype)))() `sparse_jacobian` (complete) time: $(t₁)s"

                J = SparseDiffTools.__init_𝒥(cache)

                sparse_jacobian!(J, difftype, sd, fdiff, y, x)

                @test J ≈ J_true
                @inferred sparse_jacobian!(J, difftype, sd, fdiff, y, x)

                t₂ = @belapsed sparse_jacobian!($J, $difftype, $sd, $fdiff, $y, $x)
                @info "$(nameof(typeof(difftype)))() `sparse_jacobian!` (with matrix coloring) time: $(t₂)s"
            end
        end

        for difftype in (AutoSparseZygote(), AutoZygote())
            y = similar(x)
            cache = sparse_jacobian_cache(difftype, sd, fdiff, y, x)
            J = SparseDiffTools.__init_𝒥(cache)

            @testset "Cache & Reuse" begin
                @test_throws Exception sparse_jacobian!(J, difftype, cache, fdiff, y, x)
                @test_throws Exception sparse_jacobian(difftype, cache, fdiff, y, x)
            end

            @testset "Single Use" begin
                @test_throws Exception sparse_jacobian(difftype, sd, fdiff, y, x)
                @test_throws Exception sparse_jacobian!(J, difftype, sd, fdiff, y, x)
            end
        end
    end
end
