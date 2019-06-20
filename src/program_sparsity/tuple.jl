import Cassette: tagged_new_tuple, ContextTagged, BindingMeta, DisableHooks, nametype

const TaggedSparsity = Cassette.Context{nametype(SparsityContext),X,T,P,B,H} where H<:Union{DisableHooks, Nothing} where B<:Union{Nothing, IdDict{Module,Dict{Symbol,BindingMeta}}} where P<:Cassette.AbstractPass where X where T<:Cassette.Tag

# Details of tainted value propagation

#
# The catch-all overdub method on SparsityCtx is too general and handles
# this case by returning a single `Tainted()` value, here we fix it up to
# track the tags separately
#
#
# Tuple construction should preserve the tags of each component
#

@inline function Cassette.overdub(ctx::TaggedSparsity,
                                  ::Core.Typeof(Core.tuple), args...)
    tagged_new_tuple(ctx, args...)
end

# TODO: figure out why this is necessary and the previous method isn't enough
function Cassette.overdub(ctx::SparsityContext, f::typeof(iterate), x::Tagged)
    if ismetatype(x, ctx, ProvinanceSet)
        tagged_new_tuple(ctx, x, nothing)
    else
        Cassette.recurse(ctx, f, x)
    end
end

function Cassette.overdub(ctx::SparsityContext, f::typeof(iterate), x::Tagged, st)
    if ismetatype(x, ctx, ProvinanceSet)
        nothing
    else
        Cassette.recurse(ctx, f, x, st)
    end
end
