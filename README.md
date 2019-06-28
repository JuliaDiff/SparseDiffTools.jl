# SparseDiffTools.jl

[![Join the chat at https://gitter.im/JuliaDiffEq/Lobby](https://badges.gitter.im/JuliaDiffEq/Lobby.svg)](https://gitter.im/JuliaDiffEq/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/JuliaDiffEq/SparseDiffTools.jl.svg?branch=master)](https://travis-ci.org/JuliaDiffEq/SparseDiffTools.jl)

This package is for exploiting sparsity in Jacobians and Hessians to accelerate
computations. Matrix-free Jacobian-vector product and Hessian-vector product
operators are provided that are compatible with AbstractMatrix-based libraries
like IterativeSolvers.jl for easy and efficient Newton-Krylov implementation.
Automatic and numerical differentiation are utilized and optional. In addition,
the ability to automatically detect the sparsity of a function, perform matrix
coloring, and utilize coloring in Jacobian and Hessian construction is provided.

## Example

Suppose we had the function

```julia
fcalls = 0
function f(dx,x)
  global fcalls += 1
  for i in 2:length(x)-1
    dx[i] = x[i-1] - 2x[i] + x[i+1]
  end
  dx[1] = -2x[1] + x[2]
  dx[end] = x[end-1] - 2x[end]
  nothing
end
```

For this function, we know that the sparsity pattern of the Jacobian is a
`Tridiagonal` matrix. We represent our sparsity by that matrix:

```julia
sparsity_pattern = Tridiagonal(ones(29),ones(30),ones(29))
```

Now we call `matrix_colors` to get the color vector for that matrix:

```julia
colors = matrix_colors(sparsity_pattern)
```

Since `maximum(colors)` is 3, this means that finite differencing can now
compute the Jacobian in just 4 `f`-evaluations:

```julia
J = DiffEqDiffTools.finite_difference_jacobian(f, rand(30), color=colors)
@show fcalls # 4
```

In addition, a faster forward-mode autodiff call can be utilized as well:

```julia
forwarddiff_color_jacobian!(sparsity_pattern, f, x, color = colors)
```

If one only need to compute products, one can use the operators. For example,

```julia
u = rand(30)
J = JacVec(f,u)
```

makes `J` into a matrix-free operator which calculates `J*v` products. For
example:

```julia
v = rand(30)
res = similar(v)
mul!(res,J,v) # Does 1 f evaluation
```

makes `res = J*v`. Additional operators for `HesVec` exists, including
`HesVecGrad` which allows one to utilize a gradient function. These operators
are compatible with iterative solver libraries like IterativeSolvers.jl, meaning
the following performs the Newton-Krylov update iteration:

```julia
using IterativeSolvers
gmres!(res,J,v)
```

## Documentation

### Matrix Coloring

Matrix coloring allows you to reduce the number of times finite differencing
requires an `f` call to `maximum(colors)+1`, or reduces automatic differentiation
to using `maximum(colors)` partials. Since normally these values are `length(x)`,
this can be significant savings.

The API for computing the color vector is:

```julia
matrix_colors(A::AbstractMatrix,alg::ColoringAlgorithm = GreedyD1Color(); partition_by_rows::Bool = false)
```

The first argument is the abstract matrix which represents the sparsity pattern
of the Jacobian. The second argument is the optional choice of coloring algorithm.
It will default to a greedy distance 1 coloring, though if your special matrix
type has more information, like is a `Tridiagonal` or `BlockBandedMatrix`, the
color vector will be analytically calculated instead. The variable argument
`partition_by_rows` allows you to partition the Jacobian on the basis of rows instead
of columns and generate a corresponding coloring vector which can be used for
reverse-mode AD. Default value is false.

The result is a vector which assigns a color to each column (or row) of the matrix.

### Color-Assisted Differentiation

Color-assisted differentiation for numerical differentiation is provided by
DiffEqDiffTools.jl and for automatic differentiation is provided by
ForwardDiff.jl.

For DiffEqDiffTools.jl, one simply has to use the provided `color` keyword
argument. See [the DiffEqDiffTools Jacobian documentation](https://github.com/JuliaDiffEq/DiffEqDiffTools.jl#jacobians) for more details.

For forward-mode automatic differentiation, use of a color vector is provided
by the following function:

```julia
forwarddiff_color_jacobian!(J::AbstractMatrix{<:Number},
                            f,
                            x::AbstractArray{<:Number};
                            dx = nothing,
                            color = eachindex(x))
```

This call wiil allocate the cache variables each time. To avoid allocating the
cache, construct the cache in advance:

```julia
ForwardColorJacCache(f,x,_chunksize = nothing;
                              dx = nothing,
                              color=1:length(x))
```

and utilize the following signature:

```julia
forwarddiff_color_jacobian!(J::AbstractMatrix{<:Number},
                            f,
                            x::AbstractArray{<:Number},
                            jac_cache::ForwardColorJacCache)
```

### Jacobian-Vector and Hessian-Vector Products

Matrix-free implementations of Jacobian-Vector and Hessian-Vector products is
provided in both an operator and function form. For the functions, each choice
has the choice of being in-place and out-of-place, and the in-place versions
have the ability to pass in cache vectors to be non-allocating. When in-place
the function signature for Jacobians is `f!(du,u)`, while out-of-place has
`du=f(u)`. For Hessians, all functions must be `f(u)` which returns a scalar

The functions for Jacobians are:

```julia
auto_jacvec!(du, f, x, v,
                      cache1 = ForwardDiff.Dual{DeivVecTag}.(x, v),
                      cache2 = ForwardDiff.Dual{DeivVecTag}.(x, v))

auto_jacvec(f, x, v)

# If compute_f0 is false, then `f(cache1,x)` will be computed
num_jacvec!(du,f,x,v,cache1 = similar(v),
                     cache2 = similar(v);
                     compute_f0 = true)
num_jacvec(f,x,v,f0=nothing)
```

For Hessians, the following are provided:

```julia
num_hesvec!(du,f,x,v,
             cache1 = similar(v),
             cache2 = similar(v),
             cache3 = similar(v))

num_hesvec(f,x,v)

numauto_hesvec!(du,f,x,v,
                 cache = ForwardDiff.GradientConfig(f,v),
                 cache1 = similar(v),
                 cache2 = similar(v))

numauto_hesvec(f,x,v)

autonum_hesvec!(du,f,x,v,
                 cache1 = similar(v),
                 cache2 = ForwardDiff.Dual{DeivVecTag}.(x, v),
                 cache3 = ForwardDiff.Dual{DeivVecTag}.(x, v))

autonum_hesvec(f,x,v)


numback_hesvec!(du,f,x,v,
                     cache1 = similar(v),
                     cache2 = similar(v))

numback_hesvec(f,x,v)

# Currently errors! See https://github.com/FluxML/Zygote.jl/issues/241
autoback_hesvec!(du,f,x,v,
                     cache2 = ForwardDiff.Dual{DeivVecTag}.(x, v),
                     cache3 = ForwardDiff.Dual{DeivVecTag}.(x, v))

autoback_hesvec(f,x,v)
```

`numauto` and `autonum` both mix numerical and automatic differentiation, with
the former almost always being more efficient and is thus recommended. `numback` and
`autoback` methods are numerical/ForwardDiff over reverse mode automatic differentiation
respectively, where the reverse-mode AD is provided by Zygote.jl. Currently these methods
are not competitive against `numauto`, but as Zygote.jl gets optimized these will likely
be the fastest.

In addition,
the following forms allow you to provide a gradient function `g(dx,x)` or `dx=g(x)`
respectively:

```julia
num_hesvecgrad!(du,g,x,v,
                     cache2 = similar(v),
                     cache3 = similar(v))

num_hesvecgrad(g,x,v)

auto_hesvecgrad!(du,g,x,v,
                     cache2 = ForwardDiff.Dual{DeivVecTag}.(x, v),
                     cache3 = ForwardDiff.Dual{DeivVecTag}.(x, v))

auto_hesvecgrad(g,x,v)
```

#### J*v and H*v Operators

The following produce matrix-free operators which are used for calculating
Jacobian-vector and Hessian-vector products where the differentiation takes
place at the vector `u`:

```julia
JacVec(f,u::AbstractArray;autodiff=true)
HesVec(f,u::AbstractArray;autodiff=true)
HesVecGrad(g,u::AbstractArray;autodiff=false)
```

These all have the same interface, where `J*v` utilizes the out-of-place
Jacobian-vector or Hessian-vector function, whereas `mul!(res,J,v)` utilizes
the appropriate in-place versions. To update the location of differentiation
in the operator, simply mutate the vector `u`: `J.u .= ...`.
