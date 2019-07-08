import Base.Broadcast: broadcasted

let
    # Should this be Array{Tainted,1} ?
    @test jactestval((Y,X) -> typeof(X), [1], [2]) == Array{Int, 1}
    @test jactestval((Y,X) -> typeof(Y), [1], [2]) == Array{Int, 1}

    @test jactestval((Y,X) -> eltype(X), [1], [2]) == Int
    @test jactestval((Y,X) -> typeof(X[1]), [1], [2]) == Int
    @test jactestval((Y,X) -> typeof(X[1]/1), [1], [2]) == Float64

    f(y,x) = x .+ 1

    @test jactestval((Y,X) -> broadcasted(+, X, 1)[1], [1], [2]) == 3
    @test jactestval(f, [1], [2]) == [3]
    @test sparse(jactestmeta(f, [1], [2])) == sparse([], [], true, 1, 1)

    g(y,x) = y[:] .= x .+ 1
    #g(y,x) = y .= x .+ 1 -- memove

    @test sparse(jactestmeta(g, [1], [2])) == sparse([1], [1], true)
end
