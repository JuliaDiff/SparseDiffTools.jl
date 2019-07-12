# Don't taint the value enclosed by Fixed
struct Fixed
    val
end

# Get the type of the metadata attached to a value
function metatype(x, ctx)
    if  istagged(x, ctx) && hasmetadata(x, ctx)
        typeof(metadata(x, ctx))
    else
        Cassette.NoMetaData
    end
end

macro reroute(ctx, f, g)
    quote
        function Cassette.overdub(ctx::$ctx,
                                  f::typeof($(esc(f))),
                                  args...)
            Cassette.overdub(
                ctx,
                invoke,
                $(esc(g.args[1])),
                $(esc(:(Tuple{$(g.args[2:end]...)}))),
                args...)
        end
    end
end
