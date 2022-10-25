using Documenter, SparseDiffTools

include("pages.jl")

makedocs(sitename = "SparseDiffTools.jl",
         authors = "Chris Rackauckas",
         modules = [SparseDiffTools],
         clean = true,
         doctest = false,
         format = Documenter.HTML(assets = ["assets/favicon.ico"],
                                  canonical = "https://docs.sciml.ai/SparseDiffTools/stable/"),
         pages = pages)

deploydocs(repo = "https://github.com/JuliaDiff/SparseDiffTools.jl.git";
           push_preview = true)
