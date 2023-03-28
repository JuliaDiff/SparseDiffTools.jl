# Utilities for testing update coefficient behaviour with state-dependent (i.e. dependent on u/p/t) functions

mutable struct WrapFunc{F,U,P,T}
    func::F
    u::U
    p::P
    t::T
end

(w::WrapFunc)(u) = w.p * w.t * w.func(u) 
function (w::WrapFunc)(v, u) 
    w.func(v, u)
    lmul!(w.p * w.t, v)
end

update_coefficients(w::WrapFunc, u, p, t) = WrapFunc(w.func, u, p, t)
function update_coefficients!(w::WrapFunc, u, p, t)
    w.u = u
    w.p = p
    w.t = t
end

# Helper function for testing correct update coefficients behaviour of operators
function update_coefficients_for_test!(L, u, p, t)
    update_coefficients!(L, u, p, t)
    # Force function hiding inside L to update. Should be a null-op if previous line works correctly
    update_coefficients!(L.op.f, u, p, t) 
end