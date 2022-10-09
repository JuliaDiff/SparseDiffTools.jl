function jacobian_sparsity(func::Function, output::AbstractVector{T}, input::AbstractVector{T}; kwargs...) where {T<:Number}
    Symbolics.jacobian_sparsity(func,output,input)
end
function hessian_sparsity(func::Function, input::AbstractVector{T}; kwargs...) where {T<:Number}
    vars = map(Symbolics.variable, eachindex(input))
    expr = func(vars; kwargs...)
    Symbolics.hessian_sparsity(expr, vars)
end
