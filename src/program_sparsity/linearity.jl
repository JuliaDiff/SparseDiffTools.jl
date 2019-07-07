using SpecialFunctions
import Base.Broadcast

const constant_funcs = []

const monadic_linear = [deg2rad, +, rad2deg, transpose, -, conj]

const monadic_nonlinear = [asind, log1p, acsch, erfc, digamma, acos, asec, acosh, airybiprime, acsc, cscd, log, tand, log10, csch, asinh, airyai, abs2, gamma, lgamma, erfcx, bessely0, cosh, sin, cos, atan, cospi, cbrt, acosd, bessely1, acoth, erfcinv, erf, dawson, inv, acotd, airyaiprime, erfinv, trigamma, asecd, besselj1, exp, acot, sqrt, sind, sinpi, asech, log2, tan, invdigamma, airybi, exp10, sech, erfi, coth, asin, cotd, cosd, sinh, abs, besselj0, csc, tanh, secd, atand, sec, acscd, cot, exp2, expm1, atanh]

diadic_of_linearity(::Val{(true, true, true)}) = [+, rem2pi, -, >, isless, <, isequal, max, min, convert]
diadic_of_linearity(::Val{(true, true, false)}) = [*]
#diadic_of_linearit(::(Val{(true, false, true)}) = [besselk, hankelh2, bessely, besselj, besseli, polygamma, hankelh1]
diadic_of_linearity(::Val{(true, false, false)}) = [/]
diadic_of_linearity(::Val{(false, true, false)}) = [\]
diadic_of_linearity(::Val{(false, false, false)}) = [hypot, atan, mod, rem, lbeta, ^, beta]
diadic_of_linearity(::Val) = []

haslinearity(f, nargs) = false

# some functions strip the linearity metadata

for f in constant_funcs
    @eval begin
        haslinearity(::typeof($f), ::Val) = true
        linearity(::typeof($f), ::Val) = nothing
    end
end

# linearity of a single input function is either
# Val{true}() or Val{false}()
#
for f in monadic_linear
    @eval begin
        haslinearity(::typeof($f), ::Val{1}) = true
        linearity(::typeof($f), ::Val{1}) = Val{true}()
    end
end
# linearity of a 2-arg function is:
# Val{(linear11, linear22, linear12)}()
#
# linearIJ refers to the zeroness of d^2/dxIxJ
for f in monadic_nonlinear
    @eval begin
        haslinearity(::typeof($f), ::Val{1}) = true
        linearity(::typeof($f), ::Val{1}) = Val{false}()
    end
end

for linearity_mask = 0:2^3-1
    lin = Val{map(x->x!=0, (linearity_mask & 4,
                            linearity_mask & 2,
                            linearity_mask & 1))}()

    for f in diadic_of_linearity(lin)
        @eval begin
            haslinearity(::typeof($f), ::Val{2}) = true
            linearity(::typeof($f), ::Val{2}) = $lin
        end
    end
end
