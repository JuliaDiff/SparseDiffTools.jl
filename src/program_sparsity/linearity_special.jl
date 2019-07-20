const monadic_nonlinear_special = [airyai, airyaiprime, airybi, airybiprime, besselj0, besselj1, bessely0, bessely1, dawson, digamma, erf, erfc, erfcinv, erfi, erfinv, erfcx, gamma, invdigamma, lgamma, trigamma]

const diadic_linear_false_false_false_special = [lbeta, beta]
# const diadic_linear_true_false_true_special = [besselk, hankelh2, bessely, besselj, besseli, polygamma, hankelh1]

for f in monadic_nonlinear_special
    @eval begin
        haslinearity(::typeof($f), ::Val{1}) = true
        linearity(::typeof($f), ::Val{1}) = Val(false)
    end
end

# linearity of a 2-arg function is:
# Val((linear11, linear22, linear12))
#
# linearIJ refers to the zeroness of d^2/dxIxJ
for f in diadic_linear_false_false_false_special
    @eval begin
        haslinearity(::typeof($f), ::Val{2}) = true
        linearity(::typeof($f), ::Val{2}) = Val((false,false,false))
    end
end

# for f in diadic_linear_true_false_true_special
#     @eval begin
#         haslinearity(::typeof($f), ::Val{2}) = true
#         linearity(::typeof($f), ::Val{2}) = Val((true,false,true))
#     end
# end
