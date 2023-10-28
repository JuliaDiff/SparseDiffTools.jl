## Sparse Jacobian tests
using SparseDiffTools, Symbolics, ForwardDiff, LinearAlgebra, SparseArrays, Zygote, Enzyme
using Test

@views function fdiff(y, x) # in-place
    L = length(x)
    y[2:(L - 1)] .= x[1:(L - 2)] .- 2 .* x[2:(L - 1)] .+ x[3:L]
    y[1] = -2 * x[1] + x[2]
    y[L] = x[L - 1] - 2 * x[L]
    return nothing
end

@views function fdiff(x) # out-of-place
    L = length(x)
    y₂ = x[1:(L - 2)] .- 2 .* x[2:(L - 1)] .+ x[3:L]
    y₁ = -2x[1] + x[2]
    y₃ = x[L - 1] - 2x[L]
    return vcat(y₁, y₂, y₃)
end

x = randn(Float32, 100);

J_true = ForwardDiff.jacobian(fdiff, x);

@info "`ForwardDiff.jacobian` time: $(@elapsed(ForwardDiff.jacobian(diff, x)))s"

# SparseDiffTools High-Level API
J_sparsity = Symbolics.jacobian_sparsity(fdiff, similar(x), x);
row_colorvec = SparseDiffTools.matrix_colors(J_sparsity; partition_by_rows = true)
col_colorvec = SparseDiffTools.matrix_colors(J_sparsity; partition_by_rows = false)

SPARSITY_DETECTION_ALGS = [JacPrototypeSparsityDetection(; jac_prototype = J_sparsity),
    SymbolicsSparsityDetection(), NoSparsityDetection(), ApproximateJacobianSparsity(),
    PrecomputedJacobianColorvec(; jac_prototype = J_sparsity, row_colorvec, col_colorvec)]

@testset "High-Level API" begin
    @testset "Sparsity Detection: $(nameof(typeof(sd)))" for sd in SPARSITY_DETECTION_ALGS
        @info "Sparsity Detection: $(nameof(typeof(sd)))"
        @info "Out of Place Function"

        @testset "sparse_jacobian $(nameof(typeof(difftype))): Out of Place" for difftype in (AutoSparseZygote(),
            AutoZygote(), AutoSparseForwardDiff(), AutoForwardDiff(),
            AutoSparseForwardDiff(; chunksize = 0), AutoForwardDiff(; chunksize = 0),
            AutoSparseForwardDiff(; chunksize = 4), AutoForwardDiff(; chunksize = 4),
            AutoSparseFiniteDiff(), AutoFiniteDiff(), AutoEnzyme(), AutoSparseEnzyme())
            @testset "Cache & Reuse" begin
                cache = sparse_jacobian_cache(difftype, sd, fdiff, x)
                J = init_jacobian(cache)

                sparse_jacobian!(J, difftype, cache, fdiff, x)

                @test J ≈ J_true
                @inferred sparse_jacobian!(J, difftype, cache, fdiff, x)

                t₁ = @elapsed sparse_jacobian!(J, difftype, cache, fdiff, x)
                @info "$(nameof(typeof(difftype)))() `sparse_jacobian!` (only differentiation) time: $(t₁)s"

                J = sparse_jacobian(difftype, cache, fdiff, x)

                @test J ≈ J_true

                if !(difftype isa AutoSparseForwardDiff || difftype isa AutoForwardDiff)
                    @inferred sparse_jacobian(difftype, cache, fdiff, x)
                end

                t₂ = @elapsed sparse_jacobian(difftype, cache, fdiff, x)
                @info "$(nameof(typeof(difftype)))() `sparse_jacobian` (with matrix allocation) time: $(t₂)s"
            end

            @testset "Single Use" begin
                J = sparse_jacobian(difftype, sd, fdiff, x)

                @test J ≈ J_true
                if !(difftype isa AutoSparseForwardDiff || difftype isa AutoForwardDiff)
                    @inferred sparse_jacobian(difftype, sd, fdiff, x)
                end

                t₁ = @elapsed sparse_jacobian(difftype, sd, fdiff, x)
                @info "$(nameof(typeof(difftype)))() `sparse_jacobian` (complete) time: $(t₁)s"

                cache = sparse_jacobian_cache(difftype, sd, fdiff, x)
                J = init_jacobian(cache)

                sparse_jacobian!(J, difftype, sd, fdiff, x)

                @test J ≈ J_true
                @inferred sparse_jacobian!(J, difftype, sd, fdiff, x)

                t₂ = @elapsed sparse_jacobian!(J, difftype, sd, fdiff, x)
                @info "$(nameof(typeof(difftype)))() `sparse_jacobian!` (with matrix coloring) time: $(t₂)s"
            end
        end

        @info "Inplace Place Function"

        @testset "sparse_jacobian $(nameof(typeof(difftype))): In place" for difftype in (AutoSparseForwardDiff(),
            AutoForwardDiff(), AutoSparseForwardDiff(; chunksize = 0),
            AutoForwardDiff(; chunksize = 0), AutoSparseForwardDiff(; chunksize = 4),
            AutoForwardDiff(; chunksize = 4), AutoSparseFiniteDiff(), AutoFiniteDiff(),
            AutoEnzyme(), AutoSparseEnzyme())
            y = similar(x)
            cache = sparse_jacobian_cache(difftype, sd, fdiff, y, x)

            @testset "Cache & Reuse" begin
                J = init_jacobian(cache)
                sparse_jacobian!(J, difftype, cache, fdiff, y, x)

                @test J ≈ J_true
                @inferred sparse_jacobian!(J, difftype, cache, fdiff, y, x)

                t₁ = @elapsed sparse_jacobian!(J, difftype, cache, fdiff, y, x)
                @info "$(nameof(typeof(difftype)))() `sparse_jacobian!` (only differentiation) time: $(t₁)s"

                J = sparse_jacobian(difftype, cache, fdiff, y, x)

                @test J ≈ J_true
                if !(difftype isa AutoSparseForwardDiff || difftype isa AutoForwardDiff)
                    @inferred sparse_jacobian(difftype, cache, fdiff, y, x)
                end

                t₂ = @elapsed sparse_jacobian(difftype, cache, fdiff, y, x)
                @info "$(nameof(typeof(difftype)))() `sparse_jacobian` (with jacobian allocation) time: $(t₂)s"
            end

            @testset "Single Use" begin
                J = sparse_jacobian(difftype, sd, fdiff, y, x)

                @test J ≈ J_true
                if !(difftype isa AutoSparseForwardDiff || difftype isa AutoForwardDiff)
                    @inferred sparse_jacobian(difftype, sd, fdiff, y, x)
                end

                t₁ = @elapsed sparse_jacobian(difftype, sd, fdiff, y, x)
                @info "$(nameof(typeof(difftype)))() `sparse_jacobian` (complete) time: $(t₁)s"

                J = init_jacobian(cache)

                sparse_jacobian!(J, difftype, sd, fdiff, y, x)

                @test J ≈ J_true
                @inferred sparse_jacobian!(J, difftype, sd, fdiff, y, x)

                t₂ = @elapsed sparse_jacobian!(J, difftype, sd, fdiff, y, x)
                @info "$(nameof(typeof(difftype)))() `sparse_jacobian!` (with matrix coloring) time: $(t₂)s"
            end
        end

        @testset "sparse_jacobian $(nameof(typeof(difftype))): In place" for difftype in (AutoSparseZygote(),
            AutoZygote())
            y = similar(x)
            cache = sparse_jacobian_cache(difftype, sd, fdiff, y, x)
            J = init_jacobian(cache)

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
