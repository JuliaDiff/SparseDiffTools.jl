using Amb

"""
`abstract_run(g, ctx, overdubbed_fn, args...)`

First rewrites every if statement

```julia
if <expr>
  ...
end

as

```julia
cond = <expr>
if istainted(ctx, cond) ? @amb(true, false) : cond
  ...
end
```

Then runs `g(Cassette.overdub(ctx, overdubbed_fn, args...)`
as many times as there are available paths. i.e. `2^n` ways
where `n` is the number of tainted branch conditions.

# Examples:
```
meta = Any[]
abstract_run(ctx, f. args...) do result
    push!(meta, metadata(result, ctx))
end
# do something to merge metadata from all the runs
```
"""
function abstract_run(acc, ctx::Cassette.Context, overdub_fn, args...)
    pass_ctx = Cassette.similarcontext(ctx, pass=AbsintPass)
    @ambrun begin
        acc(Cassette.overdub(pass_ctx, overdub_fn, args...))
        @amb
    end
end

"""
`istainted(ctx, cond)`

Does `cond` have any metadata?
"""
function istainted(ctx, cond)
    error("Method needed: istainted(::$(typeof(ctx)), ::Bool)." *
          " See docs for `istainted`.")
end

_choice() = (@amb true false)

# Must return 7 exprs
function rewrite_branch(ctx, stmt, extraslot, i)
    # turn
    #   gotoifnot %p #g 
    # into
    #   %t = istainted(%p)
    #   gotoifnot %t #orig
    #   %rec = @amb true false
    #   gotoifnot %rec #orig+1 (the next statement after gotoifnot)

    exprs = Any[]
    cond = stmt.args[1]        # already an SSAValue

    # insert a check to see if SSAValue(i) isa Tainted
    istainted_ssa = Core.SSAValue(i)
    push!(exprs, :($(Expr(:nooverdub, istainted))($(Expr(:contextslot)),
                              $cond)))

    # not tainted? jump to the penultimate statement
    push!(exprs, Expr(:gotoifnot, istainted_ssa, i+5))

    # tainted? then use this_here_predicate!(SSAValue(1))
    current_pred = i+2
    push!(exprs, :($_choice()))

    # Store the interpreter-provided predicate in the slot
    push!(exprs, Expr(:(=), extraslot, SSAValue(i+2)))

    push!(exprs, Core.GotoNode(i+6))

    push!(exprs, Expr(:(=), extraslot, cond))

    # here we put in the original code
    stmt1 = copy(stmt)
    stmt.args[1] = extraslot
    push!(exprs, stmt)

    exprs
end

function rewrite_ir(ctx, ref)
    # turn
    #   <val> ? t : f
    # into
    #   istainted(<val>) ? this_here_predicate!(p) : <val> ? t : f

    ir = ref.code_info
    ir = copy(ir)

    extraslot = gensym("tmp")
    push!(ir.slotnames, extraslot)
    push!(ir.slotflags, 0x00)
    extraslot = Core.SlotNumber(length(ir.slotnames))

    Cassette.insert_statements!(ir.code, ir.codelocs,
        (stmt, i) -> Base.Meta.isexpr(stmt, :gotoifnot) ? 7 : nothing, 
        (stmt, i) -> rewrite_branch(ctx, stmt, extraslot, i))

    ir.ssavaluetypes = length(ir.code)
    # Core.Compiler.validate_code(ir)
    return ir
end

const AbsintPass = Cassette.@pass rewrite_ir
