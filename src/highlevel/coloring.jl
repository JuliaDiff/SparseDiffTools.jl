struct MatrixColoringResult{C, J, NR, NC}
    colorvec::C
    jacobian_sparsity::J
    nz_rows::NR
    nz_cols::NC
end

struct NoMatrixColoring end

# Using Non-Sparse AD / NoSparsityDetection results in NoMatrixColoring
(::NoSparsityDetection)(::AbstractADType, args...; kwargs...) = NoMatrixColoring()

## If no specialization is available, we don't perform sparsity detection
(::AbstractMaybeSparsityDetection)(::AbstractADType, args...; kws...) = NoMatrixColoring()

# Prespecified Jacobian Structure
function (alg::JacPrototypeSparsityDetection)(ad::AutoSparse, args...; kwargs...)
    J = alg.jac_prototype
    colorvec = matrix_colors(J, alg.alg;
        partition_by_rows = mode(ad) isa ReverseMode)
    (nz_rows, nz_cols) = ArrayInterface.findstructralnz(J)
    return MatrixColoringResult(colorvec, J, nz_rows, nz_cols)
end

# Prespecified Colorvecs
function (alg::PrecomputedJacobianColorvec)(ad::AutoSparse, args...; kwargs...)
    colorvec = _get_colorvec(alg, mode(ad))
    J = alg.jac_prototype
    (nz_rows, nz_cols) = ArrayInterface.findstructralnz(J)
    return MatrixColoringResult(colorvec, J, nz_rows, nz_cols)
end

# Approximate Jacobian Sparsity Detection
## Right now we hardcode it to use `ForwardDiff`
function (alg::ApproximateJacobianSparsity)(
        ad::AutoSparse, f::F, x; fx = nothing,
        kwargs...) where {F}
    if !(ad isa AutoSparse{<:AutoForwardDiff})
        if ad isa AutoSparse{<:AutoPolyesterForwardDiff}
            @warn "$(ad) is only supported if `PolyesterForwardDiff` is explicitly loaded. Using ForwardDiff instead." maxlog=1
        else
            @warn "$(ad) support for approximate jacobian not implemented. Using ForwardDiff instead." maxlog=1
        end
    end
    @unpack ntrials, rng = alg
    fx = fx === nothing ? f(x) : fx
    cfg = ForwardDiff.JacobianConfig(f, x)
    J = fill!(similar(fx, length(fx), length(x)), 0)
    J_cache = similar(J)
    x_ = similar(x)
    for _ in 1:ntrials
        randn!(rng, x_)
        ForwardDiff.jacobian!(J_cache, f, x_, cfg)
        @. J += abs(J_cache)
    end
    return (JacPrototypeSparsityDetection(; jac_prototype = sparse(J), alg.alg))(ad, f, x;
        fx, kwargs...)
end

function (alg::ApproximateJacobianSparsity)(ad::AutoSparse, f::F, fx, x;
        kwargs...) where {F}
    if !(ad isa AutoSparse{<:AutoForwardDiff})
        if ad isa AutoSparse{<:AutoPolyesterForwardDiff}
            @warn "$(ad) is only supported if `PolyesterForwardDiff` is explicitly loaded. Using ForwardDiff instead." maxlog=1
        else
            @warn "$(ad) support for approximate jacobian not implemented. Using ForwardDiff instead." maxlog=1
        end
    end
    @unpack ntrials, rng = alg
    cfg = ForwardDiff.JacobianConfig(f, fx, x)
    J = fill!(similar(fx, length(fx), length(x)), 0)
    J_cache = similar(J)
    x_ = similar(x)
    for _ in 1:ntrials
        randn!(rng, x_)
        ForwardDiff.jacobian!(J_cache, f, fx, x_, cfg)
        @. J += abs(J_cache)
    end
    return (JacPrototypeSparsityDetection(; jac_prototype = sparse(J), alg.alg))(ad, f, x;
        fx, kwargs...)
end

function (alg::ApproximateJacobianSparsity)(
        ad::AutoSparse{<:AutoFiniteDiff}, f::F, x; fx = nothing,
        kwargs...) where {F}
    @unpack ntrials, rng = alg
    fx = fx === nothing ? f(x) : fx
    cache = FiniteDiff.JacobianCache(x, fx)
    J = fill!(similar(fx, length(fx), length(x)), 0)
    x_ = similar(x)
    ε = ifelse(alg.epsilon === nothing, eps(eltype(x)) * 100, alg.epsilon)
    for _ in 1:ntrials
        randn!(rng, x_)
        J_cache = FiniteDiff.finite_difference_jacobian(f, x, cache)
        @. J += (abs(J_cache) .≥ ε)  # hedge against numerical issues
    end
    return (JacPrototypeSparsityDetection(; jac_prototype = sparse(J), alg.alg))(ad, f, x;
        fx, kwargs...)
end

function (alg::ApproximateJacobianSparsity)(ad::AutoSparse{<:AutoFiniteDiff}, f!::F, fx, x;
        kwargs...) where {F}
    @unpack ntrials, rng = alg
    cache = FiniteDiff.JacobianCache(x, fx)
    J = fill!(similar(fx, length(fx), length(x)), 0)
    J_cache = similar(J)
    x_ = similar(x)
    ε = ifelse(alg.epsilon === nothing, eps(eltype(x)) * 100, alg.epsilon)
    for _ in 1:ntrials
        randn!(rng, x_)
        FiniteDiff.finite_difference_jacobian!(J_cache, f!, x_, cache)
        @. J += (abs(J_cache) .≥ ε)  # hedge against numerical issues
    end
    return (JacPrototypeSparsityDetection(; jac_prototype = sparse(J), alg.alg))(
        ad, f!, fx,
        x; kwargs...)
end

# TODO: Heuristics to decide whether to use Sparse Differentiation or not
#       Simple Idea: Check min(max(colorvec_cols), max(colorvec_rows))
