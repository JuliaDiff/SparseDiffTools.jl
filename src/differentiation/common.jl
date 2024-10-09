mutable struct JacFunctionWrapper{iip, oop, mode, F, FU, P, T} <: Function
    f::F
    fu::FU
    p::P
    t::T
end

function SciMLOperators.update_coefficients!(L::JacFunctionWrapper{iip, oop, mode}, _,
        p, t) where {iip, oop, mode}
    mode == 1 && (L.t = t)
    mode == 2 && (L.p = p)
    return L
end
function SciMLOperators.update_coefficients(L::JacFunctionWrapper{iip, oop, mode}, _, p,
        t) where {iip, oop, mode}
    return JacFunctionWrapper{iip, oop, mode, typeof(L.f), typeof(L.fu), typeof(p),
        typeof(t)}(L.f, L.fu, p,
        t)
end

__internal_iip(::JacFunctionWrapper{iip}) where {iip} = iip
__internal_oop(::JacFunctionWrapper{iip, oop}) where {iip, oop} = oop

(f::JacFunctionWrapper{true, oop, 1})(fu, u) where {oop} = f.f(fu, u, f.p, f.t)
(f::JacFunctionWrapper{true, oop, 2})(fu, u) where {oop} = f.f(fu, u, f.p)
(f::JacFunctionWrapper{true, oop, 3})(fu, u) where {oop} = f.f(fu, u)
(f::JacFunctionWrapper{true, true, 1})(u) = f.f(u, f.p, f.t)
(f::JacFunctionWrapper{true, true, 2})(u) = f.f(u, f.p)
(f::JacFunctionWrapper{true, true, 3})(u) = f.f(u)
(f::JacFunctionWrapper{true, false, 1})(u) = (f.f(f.fu, u, f.p, f.t); copy(f.fu))
(f::JacFunctionWrapper{true, false, 2})(u) = (f.f(f.fu, u, f.p); copy(f.fu))
(f::JacFunctionWrapper{true, false, 3})(u) = (f.f(f.fu, u); copy(f.fu))

(f::JacFunctionWrapper{false, true, 1})(fu, u) = (vec(fu) .= vec(f.f(u, f.p, f.t)))
(f::JacFunctionWrapper{false, true, 2})(fu, u) = (vec(fu) .= vec(f.f(u, f.p)))
(f::JacFunctionWrapper{false, true, 3})(fu, u) = (vec(fu) .= vec(f.f(u)))
(f::JacFunctionWrapper{false, true, 1})(u) = f.f(u, f.p, f.t)
(f::JacFunctionWrapper{false, true, 2})(u) = f.f(u, f.p)
(f::JacFunctionWrapper{false, true, 3})(u) = f.f(u)

# NOTE: `use_deprecated_ordering` is a way for external libraries to update to the correct
# style. In the next release, we will drop the first check
function JacFunctionWrapper(f::F, fu_, u, p, t;
        use_deprecated_ordering::Val{deporder} = Val(true)) where {F, deporder}
    # The warning instead of error ensures a non-breaking change for users relying on an
    # undefined / undocumented feature
    fu = fu_ === nothing ? copy(u) : copy(fu_)

    if deporder
        # Check this first else we were breaking things
        # In the next breaking release, we will fix the ordering of the checks
        iip = hasmethod(f, typeof((fu, u)))
        oop = hasmethod(f, typeof((u,)))
        if iip || oop
            if p !== nothing || t !== nothing
                Base.depwarn(
                    """`p` and/or `t` provided and are not `nothing`. But we
       potentially detected `f(du, u)` or `f(u)`. This can be caused by:

       1. `f(du, u)` or `f(u)` is defined, in-which case `p` and/or `t` should not
       be supplied.
       2. `f(args...)` is defined, in which case `hasmethod` can be spurious.

       Currently, we perform the check for `f(du, u)` and `f(u)` first, but in
       future breaking releases, this check will be performed last, which means
       that if `t` is provided `f(du, u, p, t)`/`f(u, p, t)` will be given
       precedence, similarly if `p` is provided `f(du, u, p)`/`f(u, p)` will be
       given precedence.""",
                    :JacFunctionWrapper)
            end
            return JacFunctionWrapper{iip, oop, 3, F, typeof(fu), typeof(p), typeof(t)}(f,
                fu, p, t)
        end
    end

    if t !== nothing
        iip = hasmethod(f, typeof((fu, u, p, t)))
        oop = hasmethod(f, typeof((u, p, t)))
        if !iip && !oop
            throw(ArgumentError("""`p` and `t` provided but `f(u, p, t)` or `f(fu, u, p, t)`
            not defined for `f`!"""))
        end
        return JacFunctionWrapper{iip, oop, 1, F, typeof(fu), typeof(p), typeof(t)}(f,
            fu, p, t)
    elseif p !== nothing
        iip = hasmethod(f, typeof((fu, u, p)))
        oop = hasmethod(f, typeof((u, p)))
        if !iip && !oop
            throw(ArgumentError("""`p` is provided but `f(u, p)` or `f(fu, u, p)`
            not defined for `f`!"""))
        end
        return JacFunctionWrapper{iip, oop, 2, F, typeof(fu), typeof(p), typeof(t)}(f,
            fu, p, t)
    end

    if !deporder
        iip = hasmethod(f, typeof((fu, u)))
        oop = hasmethod(f, typeof((u,)))
        if !iip && !oop
            throw(ArgumentError("""`p` is provided but `f(u)` or `f(fu, u)` not defined for
            `f`!"""))
        end
        return JacFunctionWrapper{iip, oop, 3, F, typeof(fu), typeof(p), typeof(t)}(f,
            fu, p, t)
    else
        throw(ArgumentError("""Couldn't determine the function signature of `f` to
        construct a JacobianWrapper!"""))
    end
end
