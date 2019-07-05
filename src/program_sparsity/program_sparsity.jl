include("sparsity_tracker.jl")
include("hessian.jl")
include("path.jl")
include("take_all_branches.jl")

export Sparsity, sparsity, hsparsity

struct Fixed
    value
end

"""
`sparsity!(f, Y, X, args...; sparsity=Sparsity(length(X), length(Y)))`

Execute the program that figures out the sparsity pattern of
the jacobian of the function `f`.

# Arguments:
- `f`: the function
- `Y`: the output array
- `X`: the input array
- `args`: trailing arguments to `f`. They are considered subject to change, unless wrapped as `Fixed(arg)`
- `S`: (optional) the sparsity pattern

Returns a `Sparsity`
"""
function sparsity!(f!, Y, X, args...; sparsity=Sparsity(length(Y), length(X)))
    path = Path()
    ctx = SparsityContext(metadata=(sparsity, path), pass=BranchesPass)
    ctx = Cassette.enabletagging(ctx, f!)
    ctx = Cassette.disablehooks(ctx)

    while true
        Cassette.recurse(ctx,
                         f!,
                         tag(Y, ctx, Output()),
                         tag(X, ctx, Input()),
                         # TODO: make this recursive
                         map(arg -> arg isa Fixed ?
                             arg.value : tag(arg, ctx, ProvinanceSet(())), args)...)

        println("Explored path: ", path)
        alldone(path) && break
        reset!(path)
    end
    sparsity
end

function hsparsity(f, X, args...)
    ctx = HessianSparsityContext()
    ctx = Cassette.enabletagging(ctx, f)
    ctx = Cassette.disablehooks(ctx)

    val = Cassette.recurse(ctx,
                     f,
                     tag(X, ctx, Input()),
                     # TODO: make this recursive
                     map(arg -> arg isa Fixed ?
                         arg.value : tag(arg, ctx, TermCombination([[]])), args)...)

    metadata(val, ctx), untag(val, ctx)
end
