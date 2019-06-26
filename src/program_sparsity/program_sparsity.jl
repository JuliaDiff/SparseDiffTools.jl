include("sparsity_tracker.jl")
include("path.jl")
include("take_all_branches.jl")

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
                         map(arg -> arg isa Fixed ?
                             arg.value : tag(arg, ctx, ProvinanceSet(())), args)...)

        println("Explored path: ", path)
        alldone(path) && break
        reset!(path)
    end
    sparsity
end

#=
function foo(Y, X)
    if 1>X[1]
        Y[3, :] .= X
    else
        Y[2, :] .= X
    end
    Y
end

ci = @code_lowered foo(rand(3,3), [23,53,83])

@show sparsity!(foo, rand(3,3), [23,53,83])
@show rewrite_ir(0, ci)


f(y,x) = y[:] .= x
sparsity!(f, [0,0,0], [23,53,83])

=#
include("tuple.jl")
