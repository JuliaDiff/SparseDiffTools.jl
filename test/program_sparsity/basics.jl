let
    # Should this be Array{Tainted,1} ?
    @test testval((Y,X) -> typeof(X), [1], [2]) == Array{Int, 1}
    @test testval((Y,X) -> typeof(Y), [1], [2]) == Array{Int, 1}

    @test testval((Y,X) -> eltype(X), [1], [2]) == Int
    @test testval((Y,X) -> typeof(X[1]), [1], [2]) == Int
    @test testval((Y,X) -> typeof(X[1]/1), [1], [2]) == Float64

    f(y,x) = x .+ 1

    @test testval((Y,X) -> broadcasted(+, X, 1)[1], [1], [2]) == 3
    @test testval(f, [1], [2]) == [3]
    @test sparse(testmeta(f, [1], [2])[1]) == sparse([], [], true, 1, 1)

    g(y,x) = y[:] .= x .+ 1
    #g(y,x) = y .= x .+ 1 -- memove

    @test sparse(testmeta(g, [1], [2])[1]) == sparse([1], [1], true)
end
