import SciMLOperators: update_coefficients, update_coefficients!

# Wrapper function for testing update coefficient behaviour with state-dependent (i.e. dependent on u/p/t) functions

mutable struct WrapFunc{F, P, T}
    func::F
    p::P
    t::T
end

(w::WrapFunc)(u) = w.p * w.t * w.func(u)
function (w::WrapFunc)(v, u)
    w.func(v, u)
    lmul!(w.p * w.t, v)
end

update_coefficients(w::WrapFunc, u, p, t) = WrapFunc(w.func, p, t)
function update_coefficients!(w::WrapFunc, u, p, t)
    w.p = p
    w.t = t
end
