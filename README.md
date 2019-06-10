# Interpolative Separable Density Fitting for the Bethe-Salpeter Equation

The code in this repository was used in the the paper (add link to arxiv).
The software is written in julia.

## Documentation

The software requires julia 1.0 to be available on the machine. It can then be installed from within the julia REPL using the command
```julia
    ]add https://github.com/fhenneke/BSE_k_ISDF.jl
```

The tests can then be run using
```julia
    ]test BSE_k_ISDF
```

The one-dimensional examples is self-contained and can be run from within the folder `examples`. The three commands
```julia
    include("errors_1d.jl")
    include("benchmarks_1d.jl")
    include("plotting_1d.jl")
```
should run the code to compute errors, benchmarks, and figures for the example in the article.

For the three-dimensional examples, additional input files have to be obtained. This can be done in two ways.

1.  Download the additional files from (include link here; ~11 GB) and add them to the `examples` folder.
2.  Generate the input files using the electronic structure code __exciting__ (add link) using the provided input files. It should be noted that some of the computations require multiple thousands of CPU hours. The version of exciting used to generate the date for the article was in development at the time of submission and has git hash (add hash of commit). The features used are expected to be included in the next major release of the __exciting__ software.

The diamond example can then be run using the commands
```julia
    include("errors_3d.jl")
    include("benchmarks_3d.jl")
    include("plotting_3d.jl")
```
