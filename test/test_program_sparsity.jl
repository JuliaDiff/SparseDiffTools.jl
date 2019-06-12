### Rules
function tester(f, Y, X, args...; sparsity=Sparsity(length(Y), length(X)))

    path = Path()
    ctx = SparsityContext(metadata=(sparsity, path), pass=BranchesPass)
    ctx = Cassette.enabletagging(ctx, f)
    ctx = Cassette.disablehooks(ctx)

    val = Cassette.overdub(ctx,
                        f,
                        tag(Y, ctx, Output()),
                        tag(X, ctx, Input()),
                        map(arg -> arg isa Fixed ? arg.value : tag(arg, ctx, ProvinanceSet(())), args)...)
    return ctx, val
end

testmeta(args...) = tester(args...)[1].metadata
testval(args...) = tester(args...) |> ((ctx,val),) -> untag(val, ctx)
testtag(args...) = tester(args...) |> ((ctx,val),) -> metadata(val, ctx)

using Test

Base.show(io::IO, ::Type{<:Cassette.Context}) = print(io, "ctx")
Base.show(io::IO, ::Type{<:Tagged}) = print(io, "tagged")

import Base.Broadcast: broadcasted, materialize, instantiate, preprocess

# Should this be Array{Tainted,1} ?
@test testval((Y,X) -> typeof(X), [1], [2]) == Array{Int64, 1}
@test testval((Y,X) -> typeof(Y), [1], [2]) == Array{Int64, 1}

@test testval((Y,X) -> eltype(X), [1], [2]) == Tainted
@test testval((Y,X) -> typeof(X[1]), [1], [2]) == Tainted
@test testval((Y,X) -> typeof(X[1]/1), [1], [2]) == Tainted

f(y,x) = x .+ 1

@test testval((Y,X) -> broadcasted(+, X, 1)[1], [1], [2]) == Tainted()
@test testval(f, [1], [2]) == [Tainted()]
@test sparse(testmeta(f, [1], [2])[1]) == sparse([], [], true, 1, 1)

g(y,x) = y .= x .+ 1

@test sparse(testmeta(g, [1], [2])[1]) == sparse([1], [1], true)
