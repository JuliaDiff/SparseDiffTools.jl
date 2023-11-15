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

function JacFunctionWrapper(f::F, fu_, u, p, t) where {F}
    # The warning instead of error ensures a non-breaking change for users relying on an
    # undefined / undocumented feature
    fu = fu_ === nothing ? copy(u) : copy(fu_)
    if t !== nothing
        iip = static_hasmethod(f, typeof((fu, u, p, t)))
        oop = static_hasmethod(f, typeof((u, p, t)))
        if !iip && !oop
            @warn """`p` and `t` provided but `f(u, p, t)` or `f(fu, u, p, t)` not defined
            for `f`! Will fallback to `f(u)` or `f(fu, u)`.""" maxlog=1
        else
            return JacFunctionWrapper{iip, oop, 1, F, typeof(fu), typeof(p), typeof(t)}(f,
                fu, p, t)
        end
    elseif p !== nothing
        iip = static_hasmethod(f, typeof((fu, u, p)))
        oop = static_hasmethod(f, typeof((u, p)))
        if !iip && !oop
            @warn """`p` provided but `f(u, p)` or `f(fu, u, p)` not defined for `f`! Will
            fallback to `f(u)` or `f(fu, u)`.""" maxlog=1
        else
            return JacFunctionWrapper{iip, oop, 2, F, typeof(fu), typeof(p), typeof(t)}(f,
                fu, p, t)
        end
    end
    iip = static_hasmethod(f, typeof((fu, u)))
    oop = static_hasmethod(f, typeof((u,)))
    !iip && !oop && throw(ArgumentError("`f(u)` or `f(fu, u)` not defined for `f`"))
    return JacFunctionWrapper{iip, oop, 3, F, typeof(fu), typeof(p), typeof(t)}(f,
        fu, p, t)
end
