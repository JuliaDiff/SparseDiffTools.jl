using LightGraphs

g0 = SimpleGraph(6)
add_edge!(g0, 1,2)
add_edge!(g0, 1,4)
add_edge!(g0, 1,5)
add_edge!(g0, 3,2)
add_edge!(g0, 3,5)
add_edge!(g0, 3,6)
add_edge!(g0, 4,5)
add_edge!(g0, 5,6)
sv0 = sort_by_degree(g0)

g1 = SimpleGraph(6)
add_edge!(g1, 2,1)
add_edge!(g1, 3,2)
add_edge!(g1, 4,2)
add_edge!(g1, 5,2)
add_edge!(g1, 6,2)
sv1 = sort_by_degree(g1)

g2 = SimpleGraph(5)
add_edge!(g2, 1,2)
add_edge!(g2, 1,3)
add_edge!(g2, 1,4)
add_edge!(g2, 4,2)
add_edge!(g2, 5,2)
add_edge!(g2, 3,4)
add_edge!(g2, 4,5)
sv2 = sort_by_degree(g2)

@testset "sort_by_degree(g)" begin
    @test sv0 = [5,1,3,2,4,6]
    @test sv1 = [2,1,3,4,5,6]
    @test sv2 = [4,1,2,3,5]
end
