using LinearAlgebra
import LinearAlgebra.BLAS

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

@reroute BLAS.dot dot(Any, Any)
@reroute BLAS.axpy! axpy!(Any,
                          AbstractArray,
                          AbstractArray)
