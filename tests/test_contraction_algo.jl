using MatrixColoring
include("test_data.jl")


for i=1:5
        out_colors = colorGraph(graphs[i])
        @test out_colors == coloring[i]
end
