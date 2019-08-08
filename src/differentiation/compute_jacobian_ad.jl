struct ForwardColorJacCache{T,T2,T3,T4,T5,T6}
    t::T
    fx::T2
    dx::T3
    p::T4
    colorvec::T5
    sparsity::T6
    chunksize::Int
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
                              colorvec=1:length(x),
                              sparsity::Union{AbstractArray,Nothing}=nothing)

    if _chunksize === nothing
        chunksize = default_chunk_size(maximum(colorvec))
    else
        chunksize = _chunksize
    end

    p = adapt.(typeof(x),generate_chunked_partials(x,colorvec,chunksize))
    t = Dual{typeof(f)}.(x,first(p))

    if dx === nothing
        fx = similar(t)
        _dx = similar(x)
    else
        fx = Dual{typeof(f)}.(dx,first(p))
        _dx = dx
    end


    ForwardColorJacCache(t,fx,_dx,p,colorvec,sparsity,getsize(chunksize))
end

generate_chunked_partials(x,colorvec,N::Integer) = generate_chunked_partials(x,colorvec,Val(N))
function generate_chunked_partials(x,colorvec,::Val{chunksize}) where chunksize

    num_of_chunks = Int(ceil(maximum(colorvec) / chunksize))

    padding_size = (chunksize - (maximum(colorvec) % chunksize)) % chunksize

    partials = BitMatrix(undef, length(x), maximum(colorvec))
    partial = BitMatrix(undef, length(x), chunksize)
    chunked_partials = Array{Array{Tuple{Vararg{Bool,chunksize}},1},1}(
                                                          undef, num_of_chunks)

    for color_i in 1:maximum(colorvec)
        for j in 1:length(x)
            partials[j,color_i] = colorvec[j]==color_i
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
                colorvec = eachindex(x),
                sparsity = ArrayInterface.has_sparsestruct(J) ? J : nothing)
    forwarddiff_color_jacobian!(J,f,x,ForwardColorJacCache(f,x,dx=dx,colorvec=colorvec,sparsity=sparsity))
end

function forwarddiff_color_jacobian!(J::AbstractMatrix{<:Number},
                f,
                x::AbstractArray{<:Number},
                jac_cache::ForwardColorJacCache)

    t = jac_cache.t
    fx = jac_cache.fx
    dx = jac_cache.dx
    p = jac_cache.p
    colorvec = jac_cache.colorvec
    sparsity = jac_cache.sparsity
    chunksize = jac_cache.chunksize
    color_i = 1
    fill!(J, zero(eltype(J)))

    if DiffEqDiffTools._use_findstructralnz(sparsity)
        rows_index, cols_index = ArrayInterface.findstructralnz(sparsity)
    else
        rows_index = nothing
        cols_index = nothing
    end

    ncols=size(J,2)

    for i in eachindex(p)
        partial_i = p[i]
        t .= Dual{typeof(f)}.(x, partial_i)
        f(fx,t)
        if !(sparsity isa Nothing)
            for j in 1:chunksize
                dx .= partials.(fx, j)
                if ArrayInterface.fast_scalar_indexing(dx)
                    DiffEqDiffTools._colorediteration!(J,sparsity,rows_index,cols_index,dx,colorvec,color_i,ncols)
                else
                    #=
                    J.nzval[rows_index] .+= (colorvec[cols_index] .== color_i) .* dx[rows_index]
                    or
                    J[rows_index, cols_index] .+= (colorvec[cols_index] .== color_i) .* dx[rows_index]
                    += means requires a zero'd out start
                    =#
                    if J isa SparseMatrixCSC
                        @. setindex!((J.nzval,),getindex((J.nzval,),rows_index) + (getindex((colorvec,),cols_index) == color_i) * getindex((dx,),rows_index),rows_index)
                    else
                        @. setindex!((J,),getindex((J,),rows_index, cols_index) + (getindex((colorvec,),cols_index) == color_i) * getindex((dx,),rows_index),rows_index, cols_index)
                    end
                end
                color_i += 1
                (color_i > maximum(colorvec)) && return
            end
        else
            for j in 1:chunksize
                col_index = (i-1)*chunksize + j
                (col_index > maximum(colorvec)) && return
                J[:, col_index] .= partials.(fx, j)
            end
        end
    end
    nothing
end
