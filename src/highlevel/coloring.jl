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
    reverse_mode = ad isa AbstractSparseReverseMode
    colorvec = matrix_colors(J, alg.alg; partition_by_rows = reverse_mode)
    (nz_rows, nz_cols) = ArrayInterface.findstructralnz(J)
    return MatrixColoringResult(colorvec, J, nz_rows, nz_cols)
end

# TODO: Heuristics to decide whether to use Sparse Differentiation or not
#       Simple Idea: Check min(max(colorvec_cols), max(colorvec_rows))