Term(i...) = TermCombination(Set([Dict(j=>1 for j in i)]))

@test htesttag(x->x, [1,2]) == Input()
@test htesttag(x->x[1], [1,2]) == Term(1)

# Tuple / struct
@test htesttag(x->(x[1],x[2])[2], [1,2]) == Term(2)

# 1-arg linear
@test htesttag(x->deg2rad(x[1]), [1,2]) == Term(1)

# 1-arg nonlinear
@test htesttag(x->sin(x[1]), [1,2]) == (Term(1) * Term(1))

# 2-arg (true,true,true)
@test htesttag(x->x[1]+x[2], [1,2]) == Term(1)+Term(2)

# 2-arg (true,true, false)
@test htesttag(x->x[1]*x[2], [1,2]) == Term(1)*Term(2)

# 2-arg (true,false,false)
@test htesttag(x->x[1]/x[2], [1,2]) == Term(1)*Term(2)*Term(2)

# 2-arg (false,true,false)
@test htesttag(x->x[1]\x[2], [1,2]) == Term(1)*Term(1)*Term(2)

# 2-arg (false,false,false) 
@test htesttag(x->hypot(x[1], x[2]), [1,2]) == (Term(1) + Term(2)) * (Term(1) + Term(2))


### Array operations

# copy
@test htesttag(x->copy(x)[1], [1,2]) == Term(1)
@test htesttag(x->x[:][1], [1,2]) == Term(1)
@test htesttag(x->x[1:1][1], [1,2]) == Term(1)

# tests `iterate`
function mysum(x)
    s = 0
    for a in x
        s += a
    end
    s
end
@test htesttag(mysum, [1,2]).terms == (Term(1) + Term(2)).terms
@test htesttag(mysum, [1,2.]).terms == (Term(1) + Term(2)).terms

using LinearAlgebra

# integer dot product falls back to generic
@test htesttag(x->dot(x,x), [1,2,3]) == sum(Term(i)*Term(i) for i=1:3)

# reroutes to generic implementation (blas.jl)
@test htesttag(x->dot(x,x), [1,2,3.]) == sum(Term(i)*Term(i) for i=1:3)
@test htesttag(x->x'x, [1,2,3.]) == sum(Term(i)*Term(i) for i=1:3)

# broadcast
@test htesttag(x->sum(x[1] .+ x[2]), [1,2,3.]) == Term(1) + Term(2)
@test htesttag(x->sum(x .+ x), [1,2,3.]) == sum(Term(i) for i=1:3)
@test htesttag(x->sum(x .* x), [1,2,3.]) == sum(Term(i)*Term(i) for i=1:3)
