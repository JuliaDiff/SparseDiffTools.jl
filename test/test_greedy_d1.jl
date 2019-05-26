using SparseDiffTools
include("test_data.jl")

for i = 1:5
    g = graphs[i]
    out_colors = greedy_d1(g)
    for v = 1:length(g.vertices)
        color = out_colors[v]
        for j in neighbors(g, v)
            if out_colors[j] == color
                 @test false
            end
        end
    end
    @test true
end
