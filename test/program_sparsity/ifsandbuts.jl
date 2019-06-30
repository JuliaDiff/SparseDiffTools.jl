let
    @test sparse((testmeta([1,2], [3,4], 3) do x,y,t
                      t>1 ? x[1] = y[2] : x[2] = y[1]
                  end)[1]) == sparse([2, 1], [1, 2], true)

    function h(y, x, t)
        if t > 0
            y[2, :] .= x
        else
            y[1, :] .= x
        end
    end

    @test sparse(testmeta(h, rand(3,3), ones(3), 3)[1]) == sparse([1,4,7,2,5,8], [1,2,3,1,2,3], true, 9, 3)
end
