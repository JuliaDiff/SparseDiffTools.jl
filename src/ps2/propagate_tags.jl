
macro proptagcontext(name)
    quote
        Cassette.@context($name)

        function Cassette.overdub(ctx::$name, f, args...)
            if any(x->x isa Tagged, args)
                val = Cassette.recurse(ctx, f, args...)
                # Inputs were tagged but the output wasn't
                if !(val isa Tagged) || metatype(val, ctx) <: Cassette.NoMetaData
                    return propagate_tags(ctx, f, val, args...)
                end
                return val
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
