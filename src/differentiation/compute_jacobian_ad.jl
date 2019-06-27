struct ForwardColorJacCache{T,T2,T3,T4,T5}
    t::T
    fx::T2
    dx::T3
    p::T4
    color::T5
end

function default_chunk_size(maxcolor)
    if maxcolor < DEFAULT_CHUNK_THRESHOLD
        Val(maxcolor)
    else
        Val(DEFAULT_CHUNK_THRESHOLD)
    end
end

getsize(::Val{N}) where N = N
getsize(N::Integer) = N

function ForwardColorJacCache(f,x,_chunksize = nothing;
                              dx = nothing,
                              color=1:length(x))

    if _chunksize === nothing
        chunksize = default_chunk_size(maximum(color))
    else
        chunksize = _chunksize
    end

    t = zeros(Dual{typeof(f), eltype(x), getsize(chunksize)},length(x))

    if dx === nothing
        fx = similar(t)
        _dx = similar(x)
    else
        fx = zeros(Dual{typeof(f), eltype(dx), getsize(chunksize)},length(dx))
        _dx = dx
    end

    p = generate_chunked_partials(x,color,chunksize)
    ForwardColorJacCache(t,fx,_dx,p,color)
end

generate_chunked_partials(x,color,N::Integer) = generate_chunked_partials(x,color,Val(N))
function generate_chunked_partials(x,color,::Val{chunksize}) where chunksize

    num_of_chunks = Int64(ceil(maximum(color) / chunksize))

    padding_size = (chunksize - (maximum(color) % chunksize)) % chunksize

    partials = BitMatrix(undef, length(x), maximum(color))
    partial = BitMatrix(undef, length(x), chunksize)
    chunked_partials = Array{Array{Tuple{Vararg{Bool,chunksize}},1},1}(
                                                          undef, num_of_chunks)

    for color_i in 1:maximum(color)
        for j in 1:length(x)
            partials[j,color_i] = color[j]==color_i
        end
    end

    padding_matrix = BitMatrix(undef, length(x), padding_size)
    partials = hcat(partials, padding_matrix)

    for i in 1:num_of_chunks
        partial[:,1] .= partials[:,(i-1)*chunksize+1]
        for j in 2:chunksize
            partial[:,j] .= partials[:,(i-1)*chunksize+j]
        end
        chunked_partials[i] = Tuple.(eachrow(partial))
    end

    chunked_partials

end

function forwarddiff_color_jacobian!(J::AbstractMatrix{<:Number},
                f,
                x::AbstractArray{<:Number};
                dx = nothing,
                color = eachindex(x))
    forwarddiff_color_jacobian!(J,f,x,ForwardColorJacCache(f,x,dx=dx,color=color))
end

function forwarddiff_color_jacobian!(J::AbstractMatrix{<:Number},
                f,
                x::AbstractArray{<:Number},
                jac_cache::ForwardColorJacCache)

    t = jac_cache.t
    fx = jac_cache.fx
    dx = jac_cache.dx
    p = jac_cache.p
    color = jac_cache.color
    color_i = 1
    chunksize = length(first(first(jac_cache.p)))

    for i in 1:length(p)
        partial_i = p[i]
        t .= Dual{typeof(f)}.(x, partial_i)
        f(fx,t)
        if J isa SparseMatrixCSC
            rows_index, cols_index, val = findnz(J)
            for j in 1:chunksize
                dx .= partials.(fx, j)
                for k in 1:length(cols_index)
                    if color[cols_index[k]] == color_i
                        J[rows_index[k], cols_index[k]] = dx[rows_index[k]]
                    end
                end
                color_i += 1
                (color_i > maximum(color)) && continue
            end
        else
            for j in 1:chunksize
                col_index = (i-1)*chunksize + j
                (col_index > maximum(color)) && continue
                J[:, col_index] .= partials.(fx, j)
            end
        end
    end
    nothing
end
