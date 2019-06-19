using ForwardDiff: Dual, jacobian, partials

function compute_jacobian!(J::AbstractMatrix{<:Number},
                f,
                x::AbstractArray{T};
                color=1:length(x)) where {T<:Number}

    t = zeros(Dual{typeof(f), Float64, maximum(color)},length(x))
    fx = similar(t); du = similar(x)
    partials_array = Array{T}(undef, length(x), maximum(color))

    for color_i in 1:maximum(color)
        for i in 1:length(x)
            if color[i]==color_i
                partials_array[i,color_i] = 1
            else
                partials_array[i,color_i] = 0
            end
        end
    end

    p = Tuple.(eachrow(partials_array))
    t .= Dual{typeof(f)}.(x, p)
    f(fx, t)

    if J isa SparseMatrixCSC
        rows_index, cols_index, val = findnz(J)
        for color_i in 1:maximum(color)
            du .= partials.(fx,color_i)
            for i in 1:length(cols_index)
                if color[cols_index[i]]==color_i
                    J[rows_index[i],cols_index[i]] = du[rows_index[i]]
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
