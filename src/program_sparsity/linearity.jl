const constant_funcs = []

const monadic_linear = [deg2rad, +, rad2deg, transpose, -, conj]

const monadic_nonlinear = [asind, log1p, acsch, acos, asec, acosh, acsc, cscd, log, tand, log10, csch, asinh, abs2, cosh, sin, cos, atan, cospi, cbrt, acosd, acoth, inv, acotd, asecd, exp, acot, sqrt, sind, sinpi, asech, log2, tan, exp10, sech, coth, asin, cotd, cosd, sinh, abs, csc, tanh, secd, atand, sec, acscd, cot, exp2, expm1, atanh]

const diadic_linear_true_true_true = [+, rem2pi, -, >, isless, <, isequal, max, min, convert]
const diadic_linear_true_true_false = [*]
const diadic_linear_true_false_false = [ / ]
const diadic_linear_false_true_false = [ \ ]
const diadic_linear_false_false_false = [hypot, atan, mod, rem, ^]

haslinearity(f, nargs) = false

# some functions strip the linearity metadata

for f in constant_funcs
    @eval begin
        haslinearity(::typeof($f), ::Val) = true
        linearity(::typeof($f), ::Val) = nothing
    end
end

# linearity of a single input function is either
# Val(true) or Val(false}
#
for f in monadic_linear
    @eval begin
        haslinearity(::typeof($f), ::Val{1}) = true
        linearity(::typeof($f), ::Val{1}) = Val(true)
    end
end

for f in monadic_nonlinear
    @eval begin
        haslinearity(::typeof($f), ::Val{1}) = true
        linearity(::typeof($f), ::Val{1}) = Val(false)
    end
end

# linearity of a 2-arg function is:
# Val((linear11, linear22, linear12))
#
# linearIJ refers to the zeroness of d^2/dxIxJ
for f in diadic_linear_true_true_true
    @eval begin
        haslinearity(::typeof($f), ::Val{2}) = true
        linearity(::typeof($f), ::Val{2}) = Val((true, true, true))
    end
end

for f in diadic_linear_true_true_false
    @eval begin
        haslinearity(::typeof($f), ::Val{2}) = true
        linearity(::typeof($f), ::Val{2}) = Val((true, true, false))
    end
end

for f in diadic_linear_true_false_false
    @eval begin
        haslinearity(::typeof($f), ::Val{2}) = true
        linearity(::typeof($f), ::Val{2}) = Val((true, false, false))
    end
end

for f in diadic_linear_false_true_false
    @eval begin
        haslinearity(::typeof($f), ::Val{2}) = true
        linearity(::typeof($f), ::Val{2}) = Val((false, true, false))
    end
end

for f in diadic_linear_false_false_false
    @eval begin
        haslinearity(::typeof($f), ::Val{2}) = true
        linearity(::typeof($f), ::Val{2}) = Val((false, false, false))
    end
end
