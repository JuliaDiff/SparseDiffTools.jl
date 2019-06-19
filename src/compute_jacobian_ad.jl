using ForwardDiff: Dual, jacobian, partials

function compute_jacobian!(J::AbstractMatrix{<:Number},
                f,
                x::AbstractArray{<:Number};
                color=1:length(x))

    partials_array = Array{Float64}(undef, length(x), maximum(color))
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
    t = zeros(Dual{Nothing, Float64, maximum(color)},0)
    for i in 1:length(x)
       push!(t, Dual(x[i], p[i]))
    end

    fx = similar(t)
    f(fx, t)

    rows_index, cols_index, val = findnz(J)
    for color_i in 1:maximum(color)
        du = partials.(fx,color_i)
        for i in 1:length(cols_index)
            if color[cols_index[i]]==color_i
                J[rows_index[i],cols_index[i]] = du[rows_index[i]]
            end
        end
    end

    J = Array(J)

end
