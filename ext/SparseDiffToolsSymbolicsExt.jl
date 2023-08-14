module SparseDiffToolsSymbolicsExt

using SparseDiffTools, Symbolics
import SparseDiffTools: AbstractSparseADType

function (alg::SymbolicsSparsityDetection)(ad::AbstractSparseADType, f, x; fx=nothing,
    kwargs...)
    fx = fx === nothing ? similar(f(x)) : dx
    f!(y, x) = (y .= f(x))
    J = Symbolics.jacobian_sparsity(f!, fx, x)
    _alg = JacPrototypeSparsityDetection(J, alg.alg)
    return _alg(ad, f, x; fx, kwargs...)
end

function (alg::SymbolicsSparsityDetection)(ad::AbstractSparseADType, f!, fx, x; kwargs...)
    J = Symbolics.jacobian_sparsity(f!, fx, x)
    _alg = JacPrototypeSparsityDetection(J, alg.alg)
    return _alg(ad, f!, fx, x; kwargs...)
end

end
