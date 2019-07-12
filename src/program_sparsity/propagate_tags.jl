
macro proptagcontext(name)
    quote
        Cassette.@context($name)

        function Cassette.overdub(ctx::$name, f, args...)
            # this check can be inferred (in theory)
            if any(x->x isa Tagged, args)
                # This is a slower check
                if !any(x->!(metatype(x, ctx) <: Cassette.NoMetaData), args)
                    return Cassette.recurse(ctx, f, args...)
                end
                val = Cassette.recurse(ctx, f, args...)

                # Inputs were tagged but the output wasn't
                if !(val isa Tagged)
                    return propagate_tags(ctx, f, val, args...)
                elseif metatype(val, ctx) <: Cassette.NoMetaData
                    return propagate_tags(ctx, f,
                                          val,
                                          args...)
                else
                    return val
                end
            else
                Cassette.recurse(ctx, f, args...)
            end
        end
    end |> esc
end


"""
`propagate_tags(ctx, f, result, args...)`

Called only if any of the `args` are Tagged.
must return `result` or a tagged version of `result`.
"""
function propagate_tags(ctx, f, result, args...)
    result
end
