var documenterSearchIndex = {"docs":
[{"location":"sparsedifftools/#API","page":"API","title":"API","text":"","category":"section"},{"location":"sparsedifftools/","page":"API","title":"API","text":"Modules = [SparseDiffTools]","category":"page"},{"location":"sparsedifftools/#ArrayInterface.matrix_colors","page":"API","title":"ArrayInterface.matrix_colors","text":"matrix_colors(A, alg::ColoringAlgorithm = GreedyD1Color();\n    partition_by_rows::Bool = false)\n\nReturn the colorvec vector for the matrix A using the chosen coloring algorithm. If a known analytical solution exists, that is used instead. The coloring defaults to a greedy distance-1 coloring.\n\nNote that if A isa SparseMatrixCSC, the sparsity pattern is defined by structural nonzeroes, ie includes explicitly stored zeros.\n\nIf ArrayInterface.fast_matrix_colors(A) is true, then uses ArrayInterface.matrix_colors(A) to compute the matrix colors.\n\n\n\n\n\n","category":"function"},{"location":"sparsedifftools/#SparseDiffTools.VecJac","page":"API","title":"SparseDiffTools.VecJac","text":"VecJac(f, u, [p, t]; autodiff = AutoFiniteDiff())\n\nReturns SciMLOperators.FunctionOperator which computes vector-jacobian product df/du * v.\n\nL = VecJac(f, u)\n\nL * v         # = df/du * v\nmul!(w, L, v) # = df/du * v\n\nL(v, p, t; VJP_input = w)    # = df/dw * v\nL(x, v, p, t; VJP_input = w) # = df/dw * v\n\n\n\n\n\n","category":"function"},{"location":"sparsedifftools/#SparseDiffTools._cols_by_rows-Tuple{Any, Any}","page":"API","title":"SparseDiffTools._cols_by_rows","text":"_cols_by_rows(rows_index,cols_index)\n\nReturns a vector of rows where each row contains a vector of its column indices.\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools._rows_by_cols-Tuple{Any, Any}","page":"API","title":"SparseDiffTools._rows_by_cols","text":"_rows_by_cols(rows_index,cols_index)\n\nReturns a vector of columns where each column contains a vector of its row indices.\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.color_graph-Tuple{Graphs.AbstractGraph, AcyclicColoring}","page":"API","title":"SparseDiffTools.color_graph","text":"color_graph(g::Graphs.AbstractGraphs, ::AcyclicColoring)\n\nReturns a coloring vector following the acyclic coloring rules (1) the coloring corresponds to a distance-1 coloring, and (2) vertices in every cycle of the graph are assigned at least three distinct colors. This variant of coloring is called acyclic since every subgraph induced by vertices assigned any two colors is a collection of trees—and hence is acyclic.\n\nReference: Gebremedhin AH, Manne F, Pothen A. New Acyclic and Star Coloring Algorithms with Application to Computing Hessians\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.color_graph-Tuple{Graphs.AbstractGraph, BacktrackingColor}","page":"API","title":"SparseDiffTools.color_graph","text":"color_graph(g::Graphs.AbstractGraph, ::BacktrackingColor)\n\nReturn a tight, distance-1 coloring of graph g using the minimum number of colors possible (i.e. the chromatic number of graph, χ(g))\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.color_graph-Tuple{Graphs.AbstractGraph, GreedyStar1Color}","page":"API","title":"SparseDiffTools.color_graph","text":"color_graph(g::Graphs.AbstractGraph, ::GreedyStar1Color)\n\nFind a coloring of a given input graph such that no two vertices connected by an edge have the same color using greedy approach. The number of colors used may be equal or greater than the chromatic number χ(G) of the graph.\n\nA star coloring is a special type of distance - 1  coloring, For a coloring to be called a star coloring, it must satisfy two conditions:\n\nevery pair of adjacent vertices receives distinct  colors\n\n(a distance-1 coloring)\n\nFor any vertex v, any color that leads to a two-colored path\n\ninvolving v and three other vertices  is  impermissible  for  v. In other words, every path on four vertices uses at least three colors.\n\nReference: Gebremedhin AH, Manne F, Pothen A. What color is your Jacobian? Graph coloring for computing derivatives. SIAM review. 2005;47(4):629-705.\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.color_graph-Tuple{Graphs.AbstractGraph, GreedyStar2Color}","page":"API","title":"SparseDiffTools.color_graph","text":"color_graph(g::Graphs.AbstractGraph, ::GreedyStar2Color)\n\nFind a coloring of a given input graph such that no two vertices connected by an edge have the same color using greedy approach. The number of colors used may be equal or greater than the chromatic number χ(G) of the graph.\n\nA star coloring is a special type of distance - 1  coloring, For a coloring to be called a star coloring, it must satisfy two conditions:\n\nevery pair of adjacent vertices receives distinct  colors\n\n(a distance-1 coloring)\n\nFor any vertex v, any color that leads to a two-colored path\n\ninvolving v and three other vertices  is  impermissible  for  v. In other words, every path on four vertices uses at least three colors.\n\nReference: Gebremedhin AH, Manne F, Pothen A. What color is your Jacobian? Graph coloring for computing derivatives. SIAM review. 2005;47(4):629-705.\n\nTODO: add text explaining the difference between star1 and star2\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.color_graph-Tuple{VertexSafeGraphs.VSafeGraph, ContractionColor}","page":"API","title":"SparseDiffTools.color_graph","text":"color_graph(G::VSafeGraph,::ContractionColor)\n\nFind a coloring of the graph g such that no two vertices connected by an edge have the same color.\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.color_graph-Tuple{VertexSafeGraphs.VSafeGraph, GreedyD1Color}","page":"API","title":"SparseDiffTools.color_graph","text":"color_graph(g::VSafeGraph, alg::GreedyD1Color)\n\nFind a coloring of a given input graph such that no two vertices connected by an edge have the same color using greedy approach. The number of colors used may be equal or greater than the chromatic number χ(G) of the graph.\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.contract!-Tuple{VertexSafeGraphs.VSafeGraph, Int64, Int64}","page":"API","title":"SparseDiffTools.contract!","text":"contract!(g, y, x)\n\nContract the vertex y to x, both of which belong to graph G, that is delete vertex y and join x with the neighbors of y if they are not already connected with an edge.\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.find-Tuple{Integer, Integer, Graphs.AbstractGraph, DataStructures.DisjointSets{<:Integer}}","page":"API","title":"SparseDiffTools.find","text":"find(w::Integer, x::Integer, g::Graphs.AbstractGraph,\n    two_colored_forest::DisjointSets{<:Integer})\n\nReturns the root of the disjoint set to which the edge connecting vertices w and x in the graph g belongs to\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.find_edge_index-Tuple{Integer, Integer, Graphs.AbstractGraph}","page":"API","title":"SparseDiffTools.find_edge_index","text":"find_edge(g::Graphs.AbstractGraph, v::Integer, w::Integer)\n\nReturns an integer equivalent to the index of the edge connecting the vertices v and w in the graph g\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.free_colors-Tuple{Integer, AbstractVector{<:Integer}, AbstractVector{<:Integer}, Vector{Integer}, Graphs.AbstractGraph, Integer}","page":"API","title":"SparseDiffTools.free_colors","text":"free_colors(x::Integer,\n            A::AbstractVector{<:Integer},\n            colors::AbstractVector{<:Integer},\n            F::Array{Integer,1},\n            g::Graphs.AbstractGraph,\n            opt::Integer)\n\nReturns set of free colors of x which are less than optimal chromatic number (opt)\n\nArguments:\n\nx: Vertex who's set of free colors is to be calculated A: List of vertices of graph g sorted in non-increasing order of degree colors: colors[i] stores the number of distinct colors used in the         coloring of vertices A[0], A[1]... A[i-1] F: F[i] stores the color of vertex i g: Graph to be colored opt: Current optimal number of colors to be used in the coloring of graph g\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.grow_star!-Tuple{DataStructures.DisjointSets{<:Integer}, AbstractVector{<:Tuple{Integer, Integer}}, Integer, Integer, Graphs.AbstractGraph, AbstractVector{<:Integer}}","page":"API","title":"SparseDiffTools.grow_star!","text":"grow_star!(two_colored_forest::DisjointSets{<:Integer},\n    first_neighbor::AbstractVector{<:Tuple{Integer, Integer}}, v::Integer, w::Integer,\n    g::Graphs.AbstractGraph, color::AbstractVector{<:Integer})\n\nGrow a 2-colored star after assigning a new color to the previously uncolored vertex v, by comparing it with the adjacent vertex w. Disjoint set is used to store stars in sets, which are identified through key edges present in g.\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.insert_new_tree!-Tuple{DataStructures.DisjointSets{<:Integer}, Integer, Integer, Graphs.AbstractGraph}","page":"API","title":"SparseDiffTools.insert_new_tree!","text":"insert_new_tree!(two_colored_forest::DisjointSets{<:Integer}, v::Integer,\n    w::Integer, g::Graphs.AbstractGraph\n\ncreates a new singleton set in the disjoint set 'twocoloredforest' consisting of the edge connecting v and w in the graph g\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.least_index-Tuple{AbstractVector{<:Integer}, AbstractVector{<:Integer}, Integer}","page":"API","title":"SparseDiffTools.least_index","text":"least_index(F::AbstractVector{<:Integer}, A::AbstractVector{<:Integer}, opt::Integer)\n\nReturns least index i such that color of vertex A[i] is equal to opt (optimal chromatic number)\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.length_common_neighbor-Tuple{VertexSafeGraphs.VSafeGraph, Int64, Int64}","page":"API","title":"SparseDiffTools.length_common_neighbor","text":"length_common_neighbor(g, z, x)\n\nFind the number of vertices that share an edge with both the vertices z and x belonging to the graph g.\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.matrix2graph","page":"API","title":"SparseDiffTools.matrix2graph","text":"matrix2graph(sparse_matrix, [partition_by_rows::Bool=true])\n\nA utility function to generate a graph from input sparse matrix, columns are represented with vertices and 2 vertices are connected with an edge only if the two columns are mutually orthogonal.\n\nNote that the sparsity pattern is defined by structural nonzeroes, ie includes explicitly stored zeros.\n\n\n\n\n\n","category":"function"},{"location":"sparsedifftools/#SparseDiffTools.max_degree_vertex-Tuple{VertexSafeGraphs.VSafeGraph, Vector{Int64}}","page":"API","title":"SparseDiffTools.max_degree_vertex","text":"max_degree_vertex(G, nn)\n\nFind the vertex in the group nn of vertices belonging to the graph G which has the highest degree.\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.max_degree_vertex-Tuple{VertexSafeGraphs.VSafeGraph}","page":"API","title":"SparseDiffTools.max_degree_vertex","text":"max_degree_vertex(G)\n\nFind the vertex in graph with highest degree.\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.merge_trees!-Tuple{DataStructures.DisjointSets{<:Integer}, Integer, Integer, Integer, Graphs.AbstractGraph}","page":"API","title":"SparseDiffTools.merge_trees!","text":"merge_trees!(two_colored_forest::DisjointSets{<:Integer}, v::Integer, w::Integer,\n    x::Integer, g::Graphs.AbstractGraph)\n\nSubroutine to merge trees present in the disjoint set which have a common edge.\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.min_index-Tuple{AbstractVector{<:Integer}, Integer}","page":"API","title":"SparseDiffTools.min_index","text":"min_index(forbidden_colors::AbstractVector{<:Integer}, v::Integer)\n\nReturns min{i > 0 such that forbidden_colors[i] != v}\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.non_neighbors-Tuple{VertexSafeGraphs.VSafeGraph, Integer}","page":"API","title":"SparseDiffTools.non_neighbors","text":"non_neighbors(G, x)\n\nFind the set of vertices belonging to the graph G which do not share an edge with the vertex x.\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.prevent_cycle!-Tuple{AbstractVector{<:Tuple{Integer, Integer}}, AbstractVector{<:Integer}, Integer, Integer, Integer, Graphs.AbstractGraph, DataStructures.DisjointSets{<:Integer}, AbstractVector{<:Integer}}","page":"API","title":"SparseDiffTools.prevent_cycle!","text":"prevent_cycle!(first_visit_to_tree::AbstractVector{<:Tuple{Integer, Integer}},\n    forbidden_colors::AbstractVector{<:Integer}, v::Integer, w::Integer, x::Integer,\n    g::Graphs.AbstractGraph, two_colored_forest::DisjointSets{<:Integer},\n    color::AbstractVector{<:Integer})\n\nSubroutine to avoid generation of 2-colored cycle due to coloring of vertex v, which is adjacent to vertices w and x in graph g. Disjoint set is used to store the induced 2-colored subgraphs/trees where the id of set is an integer representing an edge of graph 'g'\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.remove_higher_colors-Tuple{AbstractVector{<:Integer}, Integer}","page":"API","title":"SparseDiffTools.remove_higher_colors","text":"remove_higher_colors(U::AbstractVector{<:Integer}, opt::Integer)\n\nRemove all the colors which are greater than or equal to the opt (optimal chromatic number) from the set of colors U\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.sort_by_degree-Tuple{Graphs.AbstractGraph}","page":"API","title":"SparseDiffTools.sort_by_degree","text":"sort_by_degree(g::Graphs.AbstractGraph)\n\nReturns a list of the vertices of graph g sorted in non-increasing order of their degrees\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.sparse_jacobian!","page":"API","title":"SparseDiffTools.sparse_jacobian!","text":"sparse_jacobian!(J::AbstractMatrix, ad, cache::AbstractMaybeSparseJacobianCache, f, x)\nsparse_jacobian!(J::AbstractMatrix, ad, cache::AbstractMaybeSparseJacobianCache, f!, fx,\n    x)\n\nInplace update the matrix J with the Jacobian of f at x using the AD backend ad.\n\ncache is the cache object returned by sparse_jacobian_cache.\n\n\n\n\n\n","category":"function"},{"location":"sparsedifftools/#SparseDiffTools.sparse_jacobian!-Tuple{AbstractMatrix, ADTypes.AbstractADType, SparseDiffTools.AbstractMaybeSparsityDetection, Vararg{Any}}","page":"API","title":"SparseDiffTools.sparse_jacobian!","text":"sparse_jacobian!(J::AbstractMatrix, ad::AbstractADType, sd::AbstractSparsityDetection,\n    f, x; fx=nothing)\nsparse_jacobian!(J::AbstractMatrix, ad::AbstractADType, sd::AbstractSparsityDetection,\n    f!, fx, x)\n\nSequentially calls sparse_jacobian_cache and sparse_jacobian! to compute the Jacobian of f at x. Use this if the jacobian for f is computed exactly once. In all other cases, use sparse_jacobian_cache once to generate the cache and use sparse_jacobian! with the same cache to compute the jacobian.\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.sparse_jacobian-Tuple{ADTypes.AbstractADType, SparseDiffTools.AbstractMaybeSparseJacobianCache, Vararg{Any}}","page":"API","title":"SparseDiffTools.sparse_jacobian","text":"sparse_jacobian(ad::AbstractADType, cache::AbstractMaybeSparseJacobianCache, f, x)\nsparse_jacobian(ad::AbstractADType, cache::AbstractMaybeSparseJacobianCache, f!, fx, x)\n\nUse the sparsity detection cache for computing the sparse Jacobian. This allocates a new Jacobian at every function call\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.sparse_jacobian-Tuple{ADTypes.AbstractADType, SparseDiffTools.AbstractMaybeSparsityDetection, Vararg{Any}}","page":"API","title":"SparseDiffTools.sparse_jacobian","text":"sparse_jacobian(ad::AbstractADType, sd::AbstractMaybeSparsityDetection, f, x; fx=nothing)\nsparse_jacobian(ad::AbstractADType, sd::AbstractMaybeSparsityDetection, f!, fx, x)\n\nSequentially calls sparse_jacobian_cache and sparse_jacobian! to compute the Jacobian of f at x. Use this if the jacobian for f is computed exactly once. In all other cases, use sparse_jacobian_cache once to generate the cache and use sparse_jacobian! with the same cache to compute the jacobian.\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.sparse_jacobian_cache","page":"API","title":"SparseDiffTools.sparse_jacobian_cache","text":"sparse_jacobian_cache(ad::AbstractADType, sd::AbstractSparsityDetection, f, x; fx=nothing)\nsparse_jacobian_cache(ad::AbstractADType, sd::AbstractSparsityDetection, f!, fx, x)\n\nTakes the underlying AD backend ad, sparsity detection algorithm sd, function f, and input x and returns a cache object that can be used to compute the Jacobian.\n\nIf fx is not specified, it will be computed by calling f(x).\n\nReturns\n\nA cache for computing the Jacobian of type AbstractMaybeSparseJacobianCache.\n\n\n\n\n\n","category":"function"},{"location":"sparsedifftools/#SparseDiffTools.uncolor_all!-Tuple{AbstractVector{<:Integer}, AbstractVector{<:Integer}, Integer}","page":"API","title":"SparseDiffTools.uncolor_all!","text":"uncolor_all(F::AbstractVector{<:Integer}, A::AbstractVector{<:Integer}, start::Integer)\n\nUncolors all vertices A[i] where i is greater than or equal to start\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.uncolored_vertex_of_maximal_degree-Tuple{AbstractVector{<:Integer}, AbstractVector{<:Integer}}","page":"API","title":"SparseDiffTools.uncolored_vertex_of_maximal_degree","text":"uncolored_vertex_of_maximal_degree(A::AbstractVector{<:Integer},F::AbstractVector{<:Integer})\n\nReturns an uncolored vertex from the partially colored graph which has the highest degree\n\n\n\n\n\n","category":"method"},{"location":"sparsedifftools/#SparseDiffTools.vertex_degree-Tuple{VertexSafeGraphs.VSafeGraph, Int64}","page":"API","title":"SparseDiffTools.vertex_degree","text":"vertex_degree(g, z)\n\nFind the degree of the vertex z which belongs to the graph g.\n\n\n\n\n\n","category":"method"},{"location":"#SparseDiffTools.jl","page":"Home","title":"SparseDiffTools.jl","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This package is for exploiting sparsity in Jacobians and Hessians to accelerate computations. Matrix-free Jacobian-vector product and Hessian-vector product operators are provided that are compatible with AbstractMatrix-based libraries like IterativeSolvers.jl for easy and efficient Newton-Krylov implementation. It is possible to perform matrix coloring, and utilize coloring in Jacobian and Hessian construction.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Optionally, automatic and numerical differentiation are utilized.","category":"page"},{"location":"#Example","page":"Home","title":"Example","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Suppose we had the function","category":"page"},{"location":"","page":"Home","title":"Home","text":"fcalls = 0\nfunction f(y,x) # in-place\n  global fcalls += 1\n  for i in 2:length(x)-1\n    y[i] = x[i-1] - 2x[i] + x[i+1]\n  end\n  y[1] = -2x[1] + x[2]\n  y[end] = x[end-1] - 2x[end]\n  nothing\nend\n\nfunction g(x) # out-of-place\n  global fcalls += 1\n  y = zero(x)\n  for i in 2:length(x)-1\n    y[i] = x[i-1] - 2x[i] + x[i+1]\n  end\n  y[1] = -2x[1] + x[2]\n  y[end] = x[end-1] - 2x[end]\n  y\nend","category":"page"},{"location":"#High-Level-API","page":"Home","title":"High Level API","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"We need to perform the following steps to utilize SparseDiffTools:","category":"page"},{"location":"","page":"Home","title":"Home","text":"Specify a Sparsity Detection Algorithm. There are 3 possible choices currently:\nNoSparsityDetection: This will ignore any AD choice and compute the dense Jacobian\nJacPrototypeSparsityDetection: If you already know the sparsity pattern, you can specify it as JacPrototypeSparsityDetection(; jac_prototype=<sparsity pattern>).\nSymbolicsSparsityDetection: This will use Symbolics.jl to automatically detect the sparsity pattern. (Note that Symbolics.jl must be explicitly loaded before using this functionality.)\nNow choose an AD backend from ADTypes.jl:\nIf using a Non *Sparse* type, then we will not use sparsity detection.\nAll other sparse AD types will internally compute the proper sparsity pattern, and try to exploit that.\nNow there are 2 options:\nPrecompute the cache using sparse_jacobian_cache and use the sparse_jacobian or sparse_jacobian! functions to compute the Jacobian. This option is recommended if you are repeatedly computing the Jacobian for the same function.\nDirectly use sparse_jacobian or sparse_jacobian! to compute the Jacobian. This option should be used if you are only computing the Jacobian once.","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Symbolics\n\nsd = SymbolicsSparsityDetection()\nadtype = AutoSparseFiniteDiff()\nx = rand(30)\ny = similar(x)\n\n# Option 1\n## OOP Function\ncache = sparse_jacobian_cache(adtype, sd, g, x; fx=y) # Passing `fx` is needed if size(y) != size(x)\nJ = sparse_jacobian(adtype, cache, g, x)\n### Or\nJ_preallocated = similar(J)\nsparse_jacobian!(J_preallocated, adtype, cache, g, x)\n\n## IIP Function\ncache = sparse_jacobian_cache(adtype, sd, f, y, x)\nJ = sparse_jacobian(adtype, cache, f, y, x)\n### Or\nJ_preallocated = similar(J)\nsparse_jacobian!(J_preallocated, adtype, cache, f, y, x)\n\n# Option 2\n## OOP Function\nJ = sparse_jacobian(adtype, sd, g, x)\n### Or\nJ_preallocated = similar(J)\nsparse_jacobian!(J_preallocated, adtype, sd, g, x)\n\n## IIP Function\nJ = sparse_jacobian(adtype, sd, f, y, x)\n### Or\nJ_preallocated = similar(J)\nsparse_jacobian!(J_preallocated, adtype, sd, f, y, x)","category":"page"},{"location":"#Lower-Level-API","page":"Home","title":"Lower Level API","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"For this function, we know that the sparsity pattern of the Jacobian is a Tridiagonal matrix. However, if we didn't know the sparsity pattern for the Jacobian, we could use the Symbolics.jacobian_sparsity function to automatically detect the sparsity pattern. We declare that the function f outputs a vector of length 30 and takes in a vector of length 30, and jacobian_sparsity returns a SparseMatrixCSC:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Symbolics\ninput = rand(30)\noutput = similar(input)\nsparsity_pattern = Symbolics.jacobian_sparsity(f,output,input)\njac = Float64.(sparsity_pattern)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Now we call matrix_colors to get the colorvec vector for that matrix:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using SparseDiffTools\ncolors = matrix_colors(jac)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Since maximum(colors) is 3, this means that finite differencing can now compute the Jacobian in just 4 f-evaluations. Generating the sparsity pattern used 1 (pseudo) f-evaluation, so the total number of times that f is called to compute the sparsity pattern plus the entire 30x30 Jacobian is 5 times:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using FiniteDiff\nFiniteDiff.finite_difference_jacobian!(jac, f, rand(30), colorvec=colors)\n@show fcalls # 5","category":"page"},{"location":"","page":"Home","title":"Home","text":"In addition, a faster forward-mode autodiff call can be utilized as well:","category":"page"},{"location":"","page":"Home","title":"Home","text":"forwarddiff_color_jacobian!(jac, f, x, colorvec = colors)","category":"page"},{"location":"","page":"Home","title":"Home","text":"If one only needs to compute products, one can use the operators. For example,","category":"page"},{"location":"","page":"Home","title":"Home","text":"x = rand(30)\nJ = JacVec(f,x)","category":"page"},{"location":"","page":"Home","title":"Home","text":"makes J into a matrix-free operator which calculates J*v products. For example:","category":"page"},{"location":"","page":"Home","title":"Home","text":"v = rand(30)\nres = similar(v)\nmul!(res,J,v) # Does 1 f evaluation","category":"page"},{"location":"","page":"Home","title":"Home","text":"makes res = J*v. Additional operators for HesVec exists, including HesVecGrad which allows one to utilize a gradient function. These operators are compatible with iterative solver libraries like IterativeSolvers.jl, meaning the following performs the Newton-Krylov update iteration:","category":"page"},{"location":"","page":"Home","title":"Home","text":"using IterativeSolvers\ngmres!(res,J,v)","category":"page"},{"location":"#Documentation","page":"Home","title":"Documentation","text":"","category":"section"},{"location":"#Matrix-Coloring","page":"Home","title":"Matrix Coloring","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"This library extends the common ArrayInterfaceCore.matrix_colors function to allow for coloring sparse matrices using graphical techniques.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Matrix coloring allows you to reduce the number of times finite differencing requires an f call to maximum(colors)+1, or reduces automatic differentiation to using maximum(colors) partials. Since normally these values are length(x), this can be significant savings.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The API for computing the colorvec vector is:","category":"page"},{"location":"","page":"Home","title":"Home","text":"matrix_colors(A::AbstractMatrix,alg::ColoringAlgorithm = GreedyD1Color();\n              partition_by_rows::Bool = false)","category":"page"},{"location":"","page":"Home","title":"Home","text":"The first argument is the abstract matrix which represents the sparsity pattern of the Jacobian. The second argument is the optional choice of coloring algorithm. It will default to a greedy distance 1 coloring, though if your special matrix type has more information, like is a Tridiagonal or BlockBandedMatrix, the colorvec vector will be analytically calculated instead. The keyword argument partition_by_rows allows you to partition the Jacobian on the basis of rows instead of columns and generate a corresponding coloring vector which can be used for reverse-mode AD. Default value is false.","category":"page"},{"location":"","page":"Home","title":"Home","text":"The result is a vector which assigns a colorvec to each column (or row) of the matrix.","category":"page"},{"location":"#Colorvec-Assisted-Differentiation","page":"Home","title":"Colorvec-Assisted Differentiation","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Colorvec-assisted differentiation for numerical differentiation is provided by FiniteDiff.jl and for automatic differentiation is provided by ForwardDiff.jl.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For FiniteDiff.jl, one simply has to use the provided colorvec keyword argument. See the FiniteDiff Jacobian documentation for more details.","category":"page"},{"location":"","page":"Home","title":"Home","text":"For forward-mode automatic differentiation, use of a colorvec vector is provided by the following function:","category":"page"},{"location":"","page":"Home","title":"Home","text":"forwarddiff_color_jacobian!(J::AbstractMatrix{<:Number},\n                            f,\n                            x::AbstractArray{<:Number};\n                            dx = nothing,\n                            colorvec = eachindex(x),\n                            sparsity = nothing)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Notice that if a sparsity pattern is not supplied then the built Jacobian will be the compressed Jacobian: sparsity must be a sparse matrix or a structured matrix (Tridiagonal, Banded, etc. conforming to the ArrayInterfaceCore.jl specs) with the appropriate sparsity pattern to allow for decompression.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This call will allocate the cache variables each time. To avoid allocating the cache, construct the cache in advance:","category":"page"},{"location":"","page":"Home","title":"Home","text":"ForwardColorJacCache(f,x,_chunksize = nothing;\n                              dx = nothing,\n                              colorvec=1:length(x),\n                              sparsity = nothing)","category":"page"},{"location":"","page":"Home","title":"Home","text":"and utilize the following signature:","category":"page"},{"location":"","page":"Home","title":"Home","text":"forwarddiff_color_jacobian!(J::AbstractMatrix{<:Number},\n                            f,\n                            x::AbstractArray{<:Number},\n                            jac_cache::ForwardColorJacCache)","category":"page"},{"location":"","page":"Home","title":"Home","text":"dx is a pre-allocated output vector which is used to declare the output size, and thus allows for specifying a non-square Jacobian.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Also, it is possible retrieve the function value via value(jac_cache) or  value!(result, jac_cache)","category":"page"},{"location":"","page":"Home","title":"Home","text":"If one is using an out-of-place function f(x), then the alternative form ca be used:","category":"page"},{"location":"","page":"Home","title":"Home","text":"jacout = forwarddiff_color_jacobian(g, x,\n                                    dx = similar(x),\n                                    colorvec = 1:length(x),\n                                    sparsity = nothing,\n                                    jac_prototype = nothing)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Note that the out-of-place form is efficient and compatible with StaticArrays. One can specify the type and shape of the output Jacobian by giving an additional jac_prototype to the out-of place forwarddiff_color_jacobian function, otherwise it will become a dense matrix. If jac_prototype and sparsity are not specified, then the oop Jacobian will assume that the function has a square Jacobian matrix. If it is not the case, please specify the shape of output by giving dx.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Similar functionality is available for Hessians, using finite differences of forward derivatives. Given a scalar function f(x), a vector value for x, and a color vector and sparsity pattern, this can be accomplished using numauto_color_hessian or its in-place form numauto_color_hessian!.","category":"page"},{"location":"","page":"Home","title":"Home","text":"H = numauto_color_hessian(f, x, colorvec, sparsity)\nnumauto_color_hessian!(H, f, x, colorvec, sparsity)","category":"page"},{"location":"","page":"Home","title":"Home","text":"To avoid unnecessary allocations every time the Hessian is computed,  construct a ForwardColorHesCache beforehand:","category":"page"},{"location":"","page":"Home","title":"Home","text":"hescache = ForwardColorHesCache(f, x, colorvec, sparsity)\nnumauto_color_hessian!(H, f, x, hescache)","category":"page"},{"location":"","page":"Home","title":"Home","text":"By default, these methods use a mix of numerical and automatic differentiation, namely by taking finite differences of gradients calculated via ForwardDiff.jl. Alternatively, if you have your own custom gradient function g!, you can specify  it as an argument to ForwardColorHesCache:","category":"page"},{"location":"","page":"Home","title":"Home","text":"hescache = ForwardColorHesCache(f, x, colorvec, sparsity, g!)","category":"page"},{"location":"","page":"Home","title":"Home","text":"Note that any user-defined gradient needs to have the signature g!(G, x), i.e. updating the gradient G in place.","category":"page"},{"location":"#Jacobian-Vector-and-Hessian-Vector-Products","page":"Home","title":"Jacobian-Vector and Hessian-Vector Products","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Matrix-free implementations of Jacobian-Vector and Hessian-Vector products is provided in both an operator and function form. For the functions, each choice has the choice of being in-place and out-of-place, and the in-place versions have the ability to pass in cache vectors to be non-allocating. When in-place the function signature for Jacobians is f!(du,u), while out-of-place has du=f(u). For Hessians, all functions must be f(u) which returns a scalar","category":"page"},{"location":"","page":"Home","title":"Home","text":"The functions for Jacobians are:","category":"page"},{"location":"","page":"Home","title":"Home","text":"auto_jacvec!(dy, f, x, v,\n                      cache1 = ForwardDiff.Dual{DeivVecTag}.(x, v),\n                      cache2 = ForwardDiff.Dual{DeivVecTag}.(x, v))\n\nauto_jacvec(f, x, v)\n\n# If compute_f0 is false, then `f(cache1,x)` will be computed\nnum_jacvec!(dy,f,x,v,cache1 = similar(v),\n                     cache2 = similar(v);\n                     compute_f0 = true)\nnum_jacvec(f,x,v,f0=nothing)","category":"page"},{"location":"","page":"Home","title":"Home","text":"For Hessians, the following are provided:","category":"page"},{"location":"","page":"Home","title":"Home","text":"num_hesvec!(dy,f,x,v,\n             cache1 = similar(v),\n             cache2 = similar(v),\n             cache3 = similar(v))\n\nnum_hesvec(f,x,v)\n\nnumauto_hesvec!(dy,f,x,v,\n                 cache = ForwardDiff.GradientConfig(f,v),\n                 cache1 = similar(v),\n                 cache2 = similar(v))\n\nnumauto_hesvec(f,x,v)\n\nautonum_hesvec!(dy,f,x,v,\n                 cache1 = similar(v),\n                 cache2 = ForwardDiff.Dual{DeivVecTag}.(x, v),\n                 cache3 = ForwardDiff.Dual{DeivVecTag}.(x, v))\n\nautonum_hesvec(f,x,v)","category":"page"},{"location":"","page":"Home","title":"Home","text":"In addition, the following forms allow you to provide a gradient function g(dy,x) or dy=g(x) respectively:","category":"page"},{"location":"","page":"Home","title":"Home","text":"num_hesvecgrad!(dy,g,x,v,\n                     cache2 = similar(v),\n                     cache3 = similar(v))\n\nnum_hesvecgrad(g,x,v)\n\nauto_hesvecgrad!(dy,g,x,v,\n                     cache2 = ForwardDiff.Dual{DeivVecTag}.(x, v),\n                     cache3 = ForwardDiff.Dual{DeivVecTag}.(x, v))\n\nauto_hesvecgrad(g,x,v)","category":"page"},{"location":"","page":"Home","title":"Home","text":"The numauto and autonum methods both mix numerical and automatic differentiation, with the former almost always being more efficient and thus being recommended.","category":"page"},{"location":"","page":"Home","title":"Home","text":"Optionally, if you load Zygote.jl, the following numback and autoback methods are available and allow numerical/ForwardDiff over reverse mode automatic differentiation respectively, where the reverse-mode AD is provided by Zygote.jl. Currently these methods are not competitive against numauto, but as Zygote.jl gets optimized these will likely be the fastest.","category":"page"},{"location":"","page":"Home","title":"Home","text":"using Zygote # Required\n\nnumback_hesvec!(dy,f,x,v,\n                     cache1 = similar(v),\n                     cache2 = similar(v))\n\nnumback_hesvec(f,x,v)\n\n# Currently errors! See https://github.com/FluxML/Zygote.jl/issues/241\nautoback_hesvec!(dy,f,x,v,\n                     cache2 = ForwardDiff.Dual{DeivVecTag}.(x, v),\n                     cache3 = ForwardDiff.Dual{DeivVecTag}.(x, v))\n\nautoback_hesvec(f,x,v)","category":"page"},{"location":"#J*v-and-H*v-Operators","page":"Home","title":"Jv and Hv Operators","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The following produce matrix-free operators which are used for calculating Jacobian-vector and Hessian-vector products where the differentiation takes place at the vector u:","category":"page"},{"location":"","page":"Home","title":"Home","text":"JacVec(f,x::AbstractArray;autodiff=true)\nHesVec(f,x::AbstractArray;autodiff=true)\nHesVecGrad(g,x::AbstractArray;autodiff=false)","category":"page"},{"location":"","page":"Home","title":"Home","text":"These all have the same interface, where J*v utilizes the out-of-place Jacobian-vector or Hessian-vector function, whereas mul!(res,J,v) utilizes the appropriate in-place versions. To update the location of differentiation in the operator, simply mutate the vector u: J.u .= ....","category":"page"}]
}
