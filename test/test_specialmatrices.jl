using SparseDiffTools
using LinearAlgebra
using BandedMatrices

n=10
dense=fill(1.,(n,n))
uptri=UpperTriangular(dense)
lotri=LowerTriangular(dense)

diagonal=Diagonal(dense)
bidiagonalU=Bidiagonal(dense,:U)
bidiagonalL=Bidiagonal(dense,:L)
tridiagonal=Tridiagonal(dense)
symtridiagonal=SymTridiagonal(dense)

banded=BandedMatrix(dense,(1,2))

@test matrix_colors(dense)==1:n
@test matrix_colors(uptri)==1:n
@test matrix_colors(lotri)==1:n

@test matrix_colors(diagonal)==fill(1,n)
@test matrix_colors(bidiagonalU)==[1,2,1,2,1,2,1,2,1,2]
@test matrix_colors(bidiagonalL)==[1,2,1,2,1,2,1,2,1,2]
@test matrix_colors(tridiagonal)==[1,2,3,1,2,3,1,2,3,1]
@test matrix_colors(symtridiagonal)==[1,2,3,1,2,3,1,2,3,1]

@test matrix_colors(banded)==[1,2,3,4,1,2,3,4,1,2]