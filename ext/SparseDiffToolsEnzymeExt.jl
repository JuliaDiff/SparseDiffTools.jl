module SparseDiffToolsEnzymeExt

import ArrayInterface: fast_scalar_indexing
import SparseDiffTools: __f̂, __maybe_copy_x, __jacobian!, __gradient, __gradient!,
                        __test_backend_loaded
# FIXME: For Enzyme we currently assume reverse mode
import ADTypes: AutoSparse, AutoEnzyme
using Enzyme

using ForwardDiff

@inline __test_backend_loaded(::Union{AutoSparse{<:AutoEnzyme}, AutoEnzyme}) = nothing

## Satisfying High-Level Interface for Sparse Jacobians
function __gradient(::Union{AutoSparse{<:AutoEnzyme}, AutoEnzyme}, f, x, cols)
    dx = zero(x)
    autodiff(Reverse, __f̂, Const(f), Duplicated(x, dx), Const(cols))
    return vec(dx)
end

function __gradient!(::Union{AutoSparse{<:AutoEnzyme}, AutoEnzyme}, f!, fx, x, cols)
    dx = zero(x)
    dfx = zero(fx)
    autodiff(Reverse, __f̂, Active, Const(f!), Duplicated(fx, dfx), Duplicated(x, dx),
        Const(cols))
    return dx
end

function __jacobian!(J::AbstractMatrix, ::Union{AutoSparse{<:AutoEnzyme}, AutoEnzyme}, f, x)
    J .= only(jacobian(
        Reverse, a -> vec(f(reshape(a, size(x)))), vec(x); n_outs = Val(size(J, 1))))
    return J
end

@views function __jacobian!(J, ad::Union{AutoSparse{<:AutoEnzyme}, AutoEnzyme}, f!, fx, x)
    # This version is slowish not sure how to do jacobians for inplace functions
    @warn "Current code for computing jacobian for inplace functions in Enzyme is slow." maxlog=1
    dfx = zero(fx)
    dx = zero(x)

    function __f_row_idx(f!, fx, x, row_idx)
        f!(fx, x)
        if fast_scalar_indexing(fx)
            return fx[row_idx]
        else
            # Avoid the GPU Arrays scalar indexing error
            return sum(selectdim(fx, 1, row_idx:row_idx))
        end
    end

    for row_idx in 1:size(J, 1)
        autodiff(Reverse, __f_row_idx, Const(f!), DuplicatedNoNeed(fx, dfx),
            Duplicated(x, dx), Const(row_idx))
        J[row_idx, :] .= dx
        fill!(dfx, 0)
        fill!(dx, 0)
    end

    return J
end

__maybe_copy_x(::Union{AutoSparse{<:AutoEnzyme}, AutoEnzyme}, x::SubArray) = copy(x)

end
