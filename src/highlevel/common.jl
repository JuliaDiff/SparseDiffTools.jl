# Sparsity Detection
abstract type AbstractMaybeSparsityDetection end
abstract type AbstractSparsityDetection <: AbstractMaybeSparsityDetection end

struct NoSparsityDetection <: AbstractMaybeSparsityDetection end

"""
    SymbolicsSparsityDetection(; alg = GreedyD1Color())

Use Symbolics to compute the sparsity pattern of the Jacobian. This requires `Symbolics.jl`
to be explicitly loaded.

## Keyword Arguments

    - `alg`: The algorithm used for computing the matrix colors

See Also: [JacPrototypeSparsityDetection](@ref), [PrecomputedJacobianColorvec](@ref)
"""
Base.@kwdef struct SymbolicsSparsityDetection{A <: ArrayInterface.ColoringAlgorithm} <:
                   AbstractSparsityDetection
    alg::A = GreedyD1Color()
end

"""
    JacPrototypeSparsityDetection(; jac_prototype, alg = GreedyD1Color())

Use a pre-specified `jac_prototype` to compute the matrix colors of the Jacobian.

## Keyword Arguments

    - `jac_prototype`: The prototype Jacobian used for computing the matrix colors
    - `alg`: The algorithm used for computing the matrix colors

See Also: [SymbolicsSparsityDetection](@ref), [PrecomputedJacobianColorvec](@ref)
"""
Base.@kwdef struct JacPrototypeSparsityDetection{
    J, A <: ArrayInterface.ColoringAlgorithm} <: AbstractSparsityDetection
    jac_prototype::J
    alg::A = GreedyD1Color()
end

"""
    PrecomputedJacobianColorvec(jac_prototype, row_colorvec, col_colorvec)

Use a pre-specified `colorvec` which can be directly used for sparse differentiation. Based
on whether a reverse mode or forward mode or finite differences is used, the corresponding
`row_colorvec` or `col_colorvec` is used. Atmost one of them can be set to `nothing`.

## Arguments

    - `jac_prototype`: The prototype Jacobian used for computing structural nonzeros
    - `row_colorvec`: The row colorvec of the Jacobian
    - `col_colorvec`: The column colorvec of the Jacobian

See Also: [SymbolicsSparsityDetection](@ref), [JacPrototypeSparsityDetection](@ref)
"""
struct PrecomputedJacobianColorvec{J, RC, CC} <: AbstractSparsityDetection
    jac_prototype::J
    row_colorvec::RC
    col_colorvec::CC
end

"""
    PrecomputedJacobianColorvec(; jac_prototype, partition_by_rows::Bool = false,
        colorvec = missing, row_colorvec = missing, col_colorvec = missing)

Use a pre-specified `colorvec` which can be directly used for sparse differentiation. Based
on whether a reverse mode or forward mode or finite differences is used, the corresponding
`row_colorvec` or `col_colorvec` is used. Atmost one of them can be set to `nothing`.

## Keyword Arguments

    - `jac_prototype`: The prototype Jacobian used for computing structural nonzeros
    - `partition_by_rows`: Whether to partition the Jacobian by rows or columns (row
      partitioning is used for reverse mode AD)
    - `colorvec`: The colorvec of the Jacobian. If `partition_by_rows` is `true` then this
      is the row colorvec, otherwise it is the column colorvec
    - `row_colorvec`: The row colorvec of the Jacobian
    - `col_colorvec`: The column colorvec of the Jacobian

See Also: [SymbolicsSparsityDetection](@ref), [JacPrototypeSparsityDetection](@ref)
"""
function PrecomputedJacobianColorvec(; jac_prototype, partition_by_rows::Bool = false,
        colorvec = missing, row_colorvec = missing, col_colorvec = missing)
    if colorvec === missing
        @assert row_colorvec !== missing||col_colorvec !== missing "Either `colorvec` or `row_colorvec` and `col_colorvec` must be specified!"
        row_colorvec = row_colorvec === missing ? nothing : row_colorvec
        col_colorvec = col_colorvec === missing ? nothing : col_colorvec
        return PrecomputedJacobianColorvec(jac_prototype, row_colorvec, col_colorvec)
    else
        @assert row_colorvec === missing&&col_colorvec === missing "Specifying `colorvec` is incompatible with specifying `row_colorvec` or `col_colorvec`!"
        row_colorvec = partition_by_rows ? colorvec : nothing
        col_colorvec = partition_by_rows ? nothing : colorvec
        return PrecomputedJacobianColorvec(jac_prototype, row_colorvec, col_colorvec)
    end
end

function _get_colorvec(alg::PrecomputedJacobianColorvec, ad)
    cvec = alg.col_colorvec
    @assert cvec!==nothing "`col_colorvec` is nothing, but Forward Mode AD or Finite Differences is being used!"
    return cvec
end

function _get_colorvec(alg::PrecomputedJacobianColorvec, ::ReverseMode)
    cvec = alg.row_colorvec
    @assert cvec!==nothing "`row_colorvec` is nothing, but Reverse Mode AD is being used!"
    return cvec
end

"""
    ApproximateJacobianSparsity(; ntrials = 5, rng = Random.default_rng(),
        epsilon = nothing, alg = GreedyD1Color())

Use `ntrials` random vectors to compute the sparsity pattern of the Jacobian. This is an
approximate method and the sparsity pattern may not be exact.

## Keyword Arguments

    - `ntrials`: The number of random vectors to use for computing the sparsity pattern
    - `rng`: The random number generator used for generating the random vectors
    - `alg`: The algorithm used for computing the matrix colors
    - `epsilon`: For Finite Differencing based Jacobian Approximations, any number smaller
      than `epsilon` is considered to be zero. If `nothing` is specified, then this value
      is calculated as `100 * eps(eltype(x))`
"""
struct ApproximateJacobianSparsity{R <: AbstractRNG, A <: ArrayInterface.ColoringAlgorithm,
    E} <: AbstractSparsityDetection
    ntrials::Int
    rng::R
    alg::A
    epsilon::E
end

function ApproximateJacobianSparsity(; ntrials::Int = 3, epsilon = nothing,
        rng::AbstractRNG = Random.default_rng(), alg = GreedyD1Color())
    return ApproximateJacobianSparsity(ntrials, rng, alg, epsilon)
end

# No one should be using this currently
Base.@kwdef struct AutoSparsityDetection{A <: ArrayInterface.ColoringAlgorithm} <:
                   AbstractSparsityDetection
    alg::A = GreedyD1Color()
end

# Function Specifications
abstract type AbstractMaybeSparseJacobianCache end

"""
    sparse_jacobian!(J::AbstractMatrix, ad, cache::AbstractMaybeSparseJacobianCache, f, x)
    sparse_jacobian!(J::AbstractMatrix, ad, cache::AbstractMaybeSparseJacobianCache, f!, fx,
        x)

Inplace update the matrix `J` with the Jacobian of `f` at `x` using the AD backend `ad`.

`cache` is the cache object returned by `sparse_jacobian_cache`.
"""
function sparse_jacobian! end

"""
    sparse_jacobian_cache(ad::AbstractADType, sd::AbstractSparsityDetection, f, x;
        fx=nothing)
    sparse_jacobian_cache(ad::AbstractADType, sd::AbstractSparsityDetection, f!, fx, x)

Takes the underlying AD backend `ad`, sparsity detection algorithm `sd`, function `f`,
and input `x` and returns a cache object that can be used to compute the Jacobian.

If `fx` is not specified, it will be computed by calling `f(x)`.

## Returns

A cache for computing the Jacobian of type `AbstractMaybeSparseJacobianCache`.
"""
function sparse_jacobian_cache(
        ad::AbstractADType, sd::AbstractMaybeSparsityDetection, f, x; fx = nothing)
    return sparse_jacobian_cache_aux(mode(ad), ad, sd, f, x; fx)
end

function sparse_jacobian_cache(
        ad::AbstractADType, sd::AbstractMaybeSparsityDetection, f!, x, fx)
    return sparse_jacobian_cache_aux(mode(ad), ad, sd, f!, x, fx)
end

function sparse_jacobian_static_array(ad, cache, f, x::SArray)
    # Not the most performant fallback
    J = init_jacobian(cache)
    sparse_jacobian!(J, ad, cache, f, MArray(x))
    return J
end

"""
    sparse_jacobian(ad::AbstractADType, sd::AbstractMaybeSparsityDetection, f, x; fx=nothing)
    sparse_jacobian(ad::AbstractADType, sd::AbstractMaybeSparsityDetection, f!, fx, x)

Sequentially calls `sparse_jacobian_cache` and `sparse_jacobian!` to compute the Jacobian of
`f` at `x`. Use this if the jacobian for `f` is computed exactly once. In all other
cases, use `sparse_jacobian_cache` once to generate the cache and use `sparse_jacobian!`
with the same cache to compute the jacobian.

If `x` is a StaticArray, then this function tries to use a non-allocating implementation for
the jacobian computation. This is possible only for a limited backends currently.
"""
function sparse_jacobian(ad::AbstractADType, sd::AbstractMaybeSparsityDetection, args...;
        kwargs...)
    cache = sparse_jacobian_cache(ad, sd, args...; kwargs...)
    J = init_jacobian(cache)
    sparse_jacobian!(J, ad, cache, args...)
    return J
end
function sparse_jacobian(ad::AbstractADType, sd::AbstractMaybeSparsityDetection, f,
        x::SArray; kwargs...)
    cache = sparse_jacobian_cache(ad, sd, f, x; kwargs...)
    return sparse_jacobian_static_array(ad, cache, f, x)
end

"""
    sparse_jacobian(ad::AbstractADType, cache::AbstractMaybeSparseJacobianCache, f, x)
    sparse_jacobian(ad::AbstractADType, cache::AbstractMaybeSparseJacobianCache, f!, fx, x)

Use the sparsity detection `cache` for computing the sparse Jacobian. This allocates a new
Jacobian at every function call.

If `x` is a StaticArray, then this function tries to use a non-allocating implementation for
the jacobian computation. This is possible only for a limited backends currently.
"""
function sparse_jacobian(ad::AbstractADType, cache::AbstractMaybeSparseJacobianCache,
        args...)
    J = init_jacobian(cache)
    sparse_jacobian!(J, ad, cache, args...)
    return J
end
function sparse_jacobian(ad::AbstractADType, cache::AbstractMaybeSparseJacobianCache, f,
        x::SArray)
    return sparse_jacobian_static_array(ad, cache, f, x)
end

"""
    sparse_jacobian!(J::AbstractMatrix, ad::AbstractADType, sd::AbstractSparsityDetection,
        f, x; fx=nothing)
    sparse_jacobian!(J::AbstractMatrix, ad::AbstractADType, sd::AbstractSparsityDetection,
        f!, fx, x)

Sequentially calls `sparse_jacobian_cache` and `sparse_jacobian!` to compute the Jacobian of
`f` at `x`. Use this if the jacobian for `f` is computed exactly once. In all other
cases, use `sparse_jacobian_cache` once to generate the cache and use `sparse_jacobian!`
with the same cache to compute the jacobian.
"""
function sparse_jacobian!(J::AbstractMatrix, ad::AbstractADType,
        sd::AbstractMaybeSparsityDetection, args...; kwargs...)
    cache = sparse_jacobian_cache(ad, sd, args...; kwargs...)
    sparse_jacobian!(J, ad, cache, args...)
    return J
end

## Internal
function __gradient end
function __gradient! end
function __jacobian! end

"""
    init_jacobian(cache::AbstractMaybeSparseJacobianCache)

Initialize the Jacobian based on the cache. Uses sparse jacobians if possible.

!!! note
    This function doesn't alias the provided jacobian prototype. It always initializes a
    fresh jacobian that can be mutated without any side effects.
"""
function init_jacobian end

# Never thought this was a useful function externally, but I ended up using it in quite a
# few places. Keeping this till I remove uses of those.
const __init_𝒥 = init_jacobian

# Misc Functions
function __chunksize(
        ::Union{AutoSparse{<:AutoForwardDiff{C}}, AutoForwardDiff{C},
            AutoSparse{<:AutoPolyesterForwardDiff{C}}, AutoPolyesterForwardDiff{C}},
        x) where {C}
    C isa ForwardDiff.Chunk && return C
    return __chunksize(Val(C), x)
end
__chunksize(::Val{nothing}, x) = __chunksize(x)
function __chunksize(::Val{C}, x) where {C}
    if C isa Integer && !(C isa Bool)
        return C ≤ 0 ? __chunksize(x) : ForwardDiff.Chunk{C}()
    else
        error("$(C)::$(typeof(C)) is not a valid chunksize!")
    end
end

__chunksize(x) = ForwardDiff.Chunk(x)
__chunksize(x::StaticArray) = ForwardDiff.Chunk{ForwardDiff.pickchunksize(prod(Size(x)))}()

function __chunksize(::Union{AutoSparse{<:AutoForwardDiff{C}}, AutoForwardDiff{C},
        AutoSparse{<:AutoPolyesterForwardDiff{C}}, AutoPolyesterForwardDiff{C}}) where {C}
    C === nothing && return nothing
    C isa Integer && !(C isa Bool) && return C ≤ 0 ? nothing : Val(C)
    return nothing
end

__f̂(f, x, idxs) = dot(vec(f(x)), idxs)

function __f̂(f!, fx, x, idxs)
    f!(fx, x)
    return dot(vec(fx), idxs)
end

@generated function __getfield(c::T, ::Val{S}) where {T, S}
    hasfield(T, S) && return :(c.$(S))
    return :(nothing)
end

"""
    init_jacobian(cache::AbstractMaybeSparseJacobianCache;
        preserve_immutable::Val = Val(false))

Initialize the Jacobian based on the cache. Uses sparse jacobians if possible.

If `preserve_immutable` is `true`, then the Jacobian returned might be immutable, this is
relevant if the inputs are immutable like `StaticArrays`.
"""
function init_jacobian(c::AbstractMaybeSparseJacobianCache;
        preserve_immutable::Val = Val(false))
    T = promote_type(eltype(c.fx), eltype(c.x))
    return init_jacobian(__getfield(c, Val(:jac_prototype)), T, c.fx, c.x;
        preserve_immutable)
end
function init_jacobian(::Nothing, ::Type{T}, fx, x; kwargs...) where {T}
    return similar(fx, T, length(fx), length(x))
end
function init_jacobian(::Nothing, ::Type{T}, fx::StaticArray, x::StaticArray;
        preserve_immutable::Val{PI} = Val(true)) where {T, PI}
    if PI
        return StaticArrays.SArray{Tuple{length(fx), length(x)}, T}(I)
    else
        return StaticArrays.MArray{Tuple{length(fx), length(x)}, T}(undef)
    end
end
function init_jacobian(J, ::Type{T}, fx, x; kwargs...) where {T}
    return similar(J, T, size(J, 1), size(J, 2))
end
init_jacobian(J::SparseMatrixCSC, ::Type{T}, fx, x; kwargs...) where {T} = T.(J)

__maybe_copy_x(_, x) = x
__maybe_copy_x(_, ::Nothing) = nothing

# Check Backend has been loaded
## We pay a small compile time cost for this, but it avoids cryptic error messages
@inline function __test_backend_loaded(ad::AbstractADType)
    error("$(ad) requires $(__backend(ad)).jl to be loaded. Please load it.")
end

@inline __backend(ad) = nothing
@inline __backend(::Union{AutoEnzyme, AutoSparse{<:AutoEnzyme}}) = :Enzyme
@inline __backend(::Union{AutoZygote, AutoSparse{<:AutoZygote}}) = :Zygote
@inline __backend(::Union{AutoForwardDiff, AutoSparse{<:AutoForwardDiff}}) = :ForwardDiff
@inline __backend(::Union{AutoPolyesterForwardDiff, AutoSparse{<:AutoPolyesterForwardDiff}}) = :PolyesterForwardDiff
@inline __backend(::Union{AutoFiniteDiff, AutoSparse{<:AutoFiniteDiff}}) = :FiniteDiff
