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

    partials_array = Vector{Array{Tuple{Bool,Bool,Bool},1}}(undef, 0)
    chunksize = getsize(default_chunk_size(maximum(color)))
    print("chunksize = $chunksize")
    num_of_passes = Int64(ceil(length(x) / chunksize))

    start_iter = 0
    end_iter = 0

    for pass_i in 1:num_of_passes
            partial = BitMatrix(undef, length(x), maximum(color))

            start_iter = (pass_i-1) * chunksize + 1
            end_iter = pass_i * chunksize

            (pass_i == num_of_passes) && (end_iter = length(x))

            for color_i in 1:maximum(color)
                for j in start_iter:end_iter
                    if color[j]==color_i
                        partial[j,color_i] = true
                    else
                        partial[j,color_i] = false
                    end
                end
            end

            p_tuple = Tuple.(eachrow(partial))
            push!(partials_array, deepcopy(p_tuple))
    end

    partials_array
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
