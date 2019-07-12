import SparseDiffTools: TermCombination, HessInput
using Test

Term(i...) = TermCombination(Set([Dict(j=>1 for j in i)]))

@test hesstesttag(x->x, [1,2]) == HessInput()
@test hesstesttag(x->x[1], [1,2]) == Term(1)

# Tuple / struct
@test hesstesttag(x->(x[1],x[2])[2], [1,2]) == Term(2)

# 1-arg linear
@test hesstesttag(x->deg2rad(x[1]), [1,2]) == Term(1)

# 1-arg nonlinear
@test hesstesttag(x->sin(x[1]), [1,2]) == (Term(1) * Term(1))

# 2-arg (true,true,true)
@test hesstesttag(x->x[1]+x[2], [1,2]) == Term(1)+Term(2)

# 2-arg (true,true, false)
@test hesstesttag(x->x[1]*x[2], [1,2]) == Term(1)*Term(2)

# 2-arg (true,false,false)
@test hesstesttag(x->x[1]/x[2], [1,2]) == Term(1)*Term(2)*Term(2)

# 2-arg (false,true,false)
@test hesstesttag(x->x[1]\x[2], [1,2]) == Term(1)*Term(1)*Term(2)

# 2-arg (false,false,false) 
@test hesstesttag(x->hypot(x[1], x[2]), [1,2]) == (Term(1) + Term(2)) * (Term(1) + Term(2))


### Array operations

# copy
@test hesstesttag(x->copy(x)[1], [1,2]) == Term(1)
@test hesstesttag(x->x[:][1], [1,2]) == Term(1)
@test hesstesttag(x->x[1:1][1], [1,2]) == Term(1)

# tests `iterate`
function mysum(x)
    s = 0
    for a in x
        s += a
    end
    s
end
@test hesstesttag(mysum, [1,2]).terms == (Term(1) + Term(2)).terms
@test hesstesttag(mysum, [1,2.]).terms == (Term(1) + Term(2)).terms

using LinearAlgebra

# integer dot product falls back to generic
@test hesstesttag(x->dot(x,x), [1,2,3]) == sum(Term(i)*Term(i) for i=1:3)

# reroutes to generic implementation (blas.jl)
@test hesstesttag(x->dot(x,x), [1,2,3.]) == sum(Term(i)*Term(i) for i=1:3)
@test hesstesttag(x->x'x, [1,2,3.]) == sum(Term(i)*Term(i) for i=1:3)

# broadcast
@test hesstesttag(x->sum(x[1] .+ x[2]), [1,2,3.]) == Term(1) + Term(2)
@test hesstesttag(x->sum(x .+ x), [1,2,3.]) == sum(Term(i) for i=1:3)
@test hesstesttag(x->sum(x .* x), [1,2,3.]) == sum(Term(i)*Term(i) for i=1:3)
