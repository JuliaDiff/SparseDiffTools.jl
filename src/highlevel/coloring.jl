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
function (alg::JacPrototypeSparsityDetection)(ad::AbstractSparseADType, args...; kwargs...)
    J = alg.jac_prototype
    colorvec = matrix_colors(J, alg.alg;
        partition_by_rows = ad isa AbstractSparseReverseMode)
    (nz_rows, nz_cols) = ArrayInterface.findstructralnz(J)
    return MatrixColoringResult(colorvec, J, nz_rows, nz_cols)
end

# Prespecified Colorvecs
function (alg::PrecomputedJacobianColorvec)(ad::AbstractSparseADType, args...; kwargs...)
    colorvec = _get_colorvec(alg, ad)
    J = alg.jac_prototype
    (nz_rows, nz_cols) = ArrayInterface.findstructralnz(J)
    return MatrixColoringResult(colorvec, J, nz_rows, nz_cols)
end

# Approximate Jacobian Sparsity Detection
## Right now we hardcode it to use `ForwardDiff`
function (alg::ApproximateJacobianSparsity)(ad::AbstractSparseADType, f, x; fx = nothing,
    kwargs...)
    @unpack ntrials, rng = alg
    fx = fx === nothing ? f(x) : fx
    J = fill!(similar(fx, length(fx), length(x)), 0)
    cfg = ForwardDiff.JacobianConfig(f, x)
    for _ in 1:ntrials
        x_ = similar(x)
        randn!(rng, x_)
        J .+= abs.(ForwardDiff.jacobian(f, x_, cfg))
    end
    return (JacPrototypeSparsityDetection(; jac_prototype = sparse(J), alg.alg))(ad, f, x;
        fx, kwargs...)
end

function (alg::ApproximateJacobianSparsity)(ad::AbstractSparseADType, f!, fx, x; kwargs...)
    @unpack ntrials, rng = alg
    cfg = ForwardDiff.JacobianConfig(f!, fx, x)
    J = fill!(similar(fx, length(fx), length(x)), 0)
    for _ in 1:ntrials
        x_ = similar(x)
        randn!(rng, x_)
        J .+= abs.(ForwardDiff.jacobian(f!, fx, x_, cfg))
    end
    return (JacPrototypeSparsityDetection(; jac_prototype = sparse(J), alg.alg))(ad, f!, fx,
        x; kwargs...)
end

# TODO: Heuristics to decide whether to use Sparse Differentiation or not
#       Simple Idea: Check min(max(colorvec_cols), max(colorvec_rows))
