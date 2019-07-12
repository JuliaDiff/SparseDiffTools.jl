struct Fixed
    value
end

"""
`sparsity!(f, Y, X, args...; sparsity=Sparsity(length(X), length(Y)), verbose=true)`

Execute the program that figures out the sparsity pattern of
the jacobian of the function `f`.

# Arguments:
- `f`: the function
- `Y`: the output array
- `X`: the input array
- `args`: trailing arguments to `f`. They are considered subject to change, unless wrapped as `Fixed(arg)`
- `S`: (optional) the sparsity pattern
- `verbose`: (optional) whether to describe the paths taken by the sparsity detection.

Returns a `Sparsity`
"""
function sparsity!(f!, Y, X, args...; sparsity=Sparsity(length(Y), length(X)),
                                      verbose = true)
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

        verbose && println("Explored path: ", path)
        alldone(path) && break
        reset!(path)
    end
    sparse(sparsity)
end

function hsparsity(f, X, args...; verbose=true)

    terms = zero(TermCombination)
    path = Path()
    while true
        ctx = HessianSparsityContext(metadata=path, pass=BranchesPass)
        ctx = Cassette.enabletagging(ctx, f)
        ctx = Cassette.disablehooks(ctx)
        val = Cassette.recurse(ctx,
                               f,
                               tag(X, ctx, Input()),
        # TODO: make this recursive
        map(arg -> arg isa Fixed ?
            arg.value : tag(arg, ctx, one(TermCombination)), args)...)
        terms += metadata(val, ctx)
        verbose && println("Explored path: ", path)
        alldone(path) && break
        reset!(path)
    end

    _sparse(terms, length(X))
end
