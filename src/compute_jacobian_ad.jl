using ForwardDiff: Dual, jacobian, partials, DEFAULT_CHUNK_THRESHOLD

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
function generate_chunked_partials(x,color,::Val{N}) where N

    # TODO: should only go up to the chunksize each time, and should
    # generate p[i] different parts, each with less than the chunksize

    partials_array = BitMatrix{Bool}(undef, length(x), maximum(color))
    for color_i in 1:maximum(color)
        for i in 1:length(x)
            if color[i]==color_i
                partials_array[i,color_i] = true
            else
                partials_array[i,color_i] = false
            end
        end
    end
    p = Tuple.(eachrow(partials_array))
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

    # TODO: Should compute on each p[i] and decompress
    t .= Dual{typeof(f)}.(x, p)
    f(fx, t)

    if J isa SparseMatrixCSC
        rows_index, cols_index, val = findnz(J)
        for color_i in 1:maximum(color)
            dx .= partials.(fx,color_i)
            for i in 1:length(cols_index)
                if color[cols_index[i]]==color_i
                    J[rows_index[i],cols_index[i]] = dx[rows_index[i]]
                end
            end
        end
    else # Compute the compressed version
        for color_i in 1:maximum(color)
            J[:,i] .= partials.(fx,color_i)
        end
    end
    nothing
end
