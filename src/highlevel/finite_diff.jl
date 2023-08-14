struct FiniteDiffJacobianCache{CO, CA, J, FX, X} <: AbstractMaybeSparseJacobianCache
    coloring::CO
    cache::CA
    jac_prototype::J
    fx::FX
    x::X
end

function sparse_jacobian_cache(fd::Union{AutoSparseFiniteDiff, AutoFiniteDiff},
    sd::AbstractMaybeSparsityDetection, f, x; fx = nothing)
    coloring_result = sd(fd, f, x)
    fx = fx === nothing ? similar(f(x)) : fx
    if coloring_result isa NoMatrixColoring
        cache = FiniteDiff.JacobianCache(x, fx)
        jac_prototype = nothing
    else
        cache = FiniteDiff.JacobianCache(x, fx; coloring_result.colorvec,
            sparsity = coloring_result.jacobian_sparsity)
        jac_prototype = coloring_result.jacobian_sparsity
    end
    return FiniteDiffJacobianCache(coloring_result, cache, jac_prototype, fx, x)
end

function sparse_jacobian_cache(fd::Union{AutoSparseFiniteDiff, AutoFiniteDiff},
    sd::AbstractMaybeSparsityDetection, f!, fx, x)
    coloring_result = sd(fd, f!, fx, x)
    if coloring_result isa NoMatrixColoring
        cache = FiniteDiff.JacobianCache(x, fx)
        jac_prototype = nothing
    else
        cache = FiniteDiff.JacobianCache(x, fx; coloring_result.colorvec,
            sparsity = coloring_result.jacobian_sparsity)
        jac_prototype = coloring_result.jacobian_sparsity
    end
    return FiniteDiffJacobianCache(coloring_result, cache, jac_prototype, fx, x)
end

function sparse_jacobian!(J::AbstractMatrix, fd, cache::FiniteDiffJacobianCache, f, x)
    f!(y, x) = (y .= f(x))
    return sparse_jacobian!(J, fd, cache, f!, cache.fx, x)
end

function sparse_jacobian!(J::AbstractMatrix, _, cache::FiniteDiffJacobianCache, f!, _, x)
    FiniteDiff.finite_difference_jacobian!(J, f!, x, cache.cache)
    return J
end
