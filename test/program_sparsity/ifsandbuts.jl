let
    @test sparse((jactestmeta([1,2], [3,4], 3) do x,y,t
                      t>1 ? x[1] = y[2] : x[2] = y[1]
                  end)) == sparse([2, 1], [1, 2], true)

    function h(y, x, t)
        if t > 0
            y[2, :] .= x
        else
            y[1, :] .= x
        end
    end

    @test sparse(jactestmeta(h, rand(3,3), ones(3), 3)) == sparse([1,4,7,2,5,8], [1,2,3,1,2,3], true, 9, 3)
end
