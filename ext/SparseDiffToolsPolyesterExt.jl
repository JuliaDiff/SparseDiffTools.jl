module SparseDiffToolsPolyesterExt

using Adapt, ArrayInterface, ForwardDiff, FiniteDiff, Polyester, SparseDiffTools,
      SparseArrays
import SparseDiffTools: polyesterforwarddiff_color_jacobian, ForwardColorJacCache,
                        __parameterless_type

function cld_fast(a::A, b::B) where {A, B}
    T = promote_type(A, B)
    return cld_fast(a % T, b % T)
end
function cld_fast(n::T, d::T) where {T}
    x = Base.udiv_int(n, d)
    x += n != d * x
    return x
end

function polyesterforwarddiff_color_jacobian(J::AbstractMatrix{<:Number}, f::F,
        x::AbstractArray{<:Number}, jac_cache::ForwardColorJacCache) where {F}
    t = jac_cache.t
    dx = jac_cache.dx
    p = jac_cache.p
    colorvec = jac_cache.colorvec
    sparsity = jac_cache.sparsity
    chunksize = jac_cache.chunksize
    maxcolor = maximum(colorvec)

    vecx = vec(x)

    nrows, ncols = size(J)

    if !(sparsity isa Nothing)
        rows_index, cols_index = ArrayInterface.findstructralnz(sparsity)
        rows_index = [rows_index[i] for i in 1:length(rows_index)]
        cols_index = [cols_index[i] for i in 1:length(cols_index)]
    else
        rows_index = 1:nrows
        cols_index = 1:ncols
    end

    if J isa AbstractSparseMatrix
        fill!(nonzeros(J), zero(eltype(J)))
    else
        fill!(J, zero(eltype(J)))
    end

    batch((length(p), min(length(p), Threads.nthreads()))) do _, start, stop
        color_i = (start - 1) * chunksize + 1
        for i in start:stop
            partial_i = p[i]
            t_ = reshape(eltype(t).(vecx, ForwardDiff.Partials.(partial_i)), size(t))
            fx = f(t_)
            for j in 1:chunksize
                dx = vec(ForwardDiff.partials.(fx, j))
                pick_inds = [idx
                             for idx in 1:length(rows_index)
                             if colorvec[cols_index[idx]] == color_i]
                rows_index_c = rows_index[pick_inds]
                cols_index_c = cols_index[pick_inds]
                @simd for i in eachindex(rows_index_c, cols_index_c)
                    J[rows_index_c[i], cols_index_c[i]] = dx[rows_index_c[i]]
                end
                color_i += 1
            end
        end
    end
    return J
end

end
