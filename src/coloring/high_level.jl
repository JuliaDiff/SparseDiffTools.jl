abstract type ColoringAlgorithm end
struct GreedyD1Color <: ColoringAlgorithm end
struct BSCColor <: ColoringAlgorithm end
struct ContractionColor <: ColoringAlgorithm end

"""
    matrix_colors(A,alg::ColoringAlgorithm = GreedyD1Color())

    Returns the color vector for the matrix A using the chosen coloring
    algorithm. If a known analytical solution exists, that is used instead.
    The coloring defaults to a greedy distance-1 coloring.

"""
function matrix_colors(A::AbstractMatrix,alg::ColoringAlgorithm = GreedyD1Color(); partition_by_rows::Bool = false)
    _A = A isa SparseMatrixCSC ? A : sparse(A) # Avoid the copy
    A_graph = matrix2graph(_A, partition_by_rows)
    color_graph(A_graph,alg)
end

"""
    matrix_colors(A::Union{Array,UpperTriangular,LowerTriangular})

    The color vector for dense matrix and triangular matrix is simply
    `[1,2,3,...,size(A,2)]`
"""
function matrix_colors(A::Union{Array,UpperTriangular,LowerTriangular})
    eachindex(1:size(A,2)) # Vector size matches number of rows
end

function _cycle(repetend,len)
    repeat(repetend,div(len,length(repetend))+1)[1:len]
end

function matrix_colors(A::Diagonal)
    fill(1,size(A,2))
end

function matrix_colors(A::Bidiagonal)
    _cycle(1:2,size(A,2))
end

function matrix_colors(A::Union{Tridiagonal,SymTridiagonal})
    _cycle(1:3,size(A,2))
end

function matrix_colors(A::BandedMatrix)
    u,l=bandwidths(A)
    width=u+l+1
    _cycle(1:width,size(A,2))
end

function matrix_colors(A::BlockBandedMatrix)
    l,u=blockbandwidths(A)
    blockwidth=l+u+1
    nblock=nblocks(A,2)
    cols=[blocksize(A,(1,i))[2] for i in 1:nblock]
    blockcolors=_cycle(1:blockwidth,nblock)
    #the reserved number of colors of a block is the maximum length of columns of blocks with the same block color
    ncolors=[maximum(cols[i:blockwidth:nblock]) for i in 1:blockwidth]
    endinds=cumsum(ncolors)
    startinds=[endinds[i]-ncolors[i]+1 for i in 1:blockwidth]
    colors=[(startinds[blockcolors[i]]:endinds[blockcolors[i]])[1:cols[i]] for i in 1:nblock]
    vcat(colors...)
end

function matrix_colors(A::BandedBlockBandedMatrix)
    l,u=blockbandwidths(A)
    lambda,mu=subblockbandwidths(A)
    blockwidth=l+u+1
    subblockwidth=lambda+mu+1
    nblock=nblocks(A,2)
    cols=[blocksize(A,(1,i))[2] for i in 1:nblock]
    blockcolors=_cycle(1:blockwidth,nblock)
    #the reserved number of colors of a block is the min of subblockwidth and the largest length of columns of blocks with the same block color
    ncolors=[min(subblockwidth,maximum(cols[i:blockwidth:nblock])) for i in 1:blockwidth]
    endinds=cumsum(ncolors)
    startinds=[endinds[i]-ncolors[i]+1 for i in 1:blockwidth]
    colors=[_cycle(startinds[blockcolors[i]]:endinds[blockcolors[i]],cols[i]) for i in 1:nblock]
    vcat(colors...)
end
