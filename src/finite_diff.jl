"""
    finite_diff(J, f, fx, coloring)

Compute the jacobian of function  f  at  given
point x via central forward differencing along
direction vectors determined by graph coloring
"""
function finite_diff(
    f,
    x::AbstractArray{<:Number},
    coloring::AbstractArray{<:Integer})

    num_sets = maximum(coloring)

    fx = f(x)
    n = length(x)
    m = length(fx)

    J = zeros(Float64, m , n)

    A = rand(n,n);
    id = diagm(0=>fill(1., size(A,1)))

    ep = sqrt(eps(real(Float64)))

    for i = 1:n
        dx = x + ep.*id[:,i]
        J[:,i] = (f(dx)-fx) / ep
    end

    return J

end


"""
        find_vectors(coloring)

Calculate the direction vectors needed for forward
difference function from the given coloring  of  a
graph
"""
function find_vectors(coloring::AbstractArray{<:Integer})

    num_sets = maximum(coloring)

    direction_vectors = [Vector{Int64}() for _ in 1:num_sets]
    for i = 1:num_sets
        dir = zeros(Int64,0)
        color = i
        for j = 1:length(coloring)
            if coloring[j] == color
                push!(dir,1)
            else
                push!(dir,0)
            end
        end
        direction_vectors[i] = copy(dir)
    end

    return direction_vectors

end
