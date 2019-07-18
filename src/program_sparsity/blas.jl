# generic implementations

macro reroute(f, g)
    quote
        function Cassette.overdub(ctx::HessianSparsityContext,
                                  f::typeof($(esc(f))),
                                  args...)
            println("rerouted")
            Cassette.overdub(
                ctx,
                invoke,
                $(esc(g.args[1])),
                $(esc(:(Tuple{$(g.args[2:end]...)}))),
                args...)
        end
    end
end

@reroute LinearAlgebra.BLAS.dot LinearAlgebra.dot(Any, Any)
@reroute LinearAlgebra.BLAS.axpy! LinearAlgebra.axpy!(Any,
                                                      AbstractArray,
                                                      AbstractArray)
