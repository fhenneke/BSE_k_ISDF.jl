# Interpolative Separable Density Fitting for the Bethe-Salpeter Equation

The code in this repository was used in the the paper

    F. Henneke, L. Lin, C. Vorwerk, C. Draxl, R. Klein, C. Yang, Fast optical absorption spectra calculations for periodic solid state systems, submitted

see [arxiv](https://arxiv.org/abs/1907.02827) for a preprint. The software is written in julia.

## Documentation

The software requires julia 1.0 to be available on the machine. It can then be installed from within the julia REPL using the command
```julia
    ]add https://github.com/fhenneke/BSE_k_ISDF.jl
```

The tests can then be run using
```julia
    ]test BSE_k_ISDF
```

To run the examples, you need to install additional dependencies via
```julia
    ]add BenchmarkTools JLD2 FileIO FFTW Arpack
```

The one-dimensional examples is self-contained and can be run from within the folder `examples`. The three commands
```julia
    include("benchmarks_1d.jl") # takes about  400 seconds
    include("errors_1d.jl")     # takes about 4000 seconds
```
should run the code to compute benchmarks and errors.

To produce the figures from the article you additionally have to use the `PGFPlotsX` package. The figures are the created using the command
```julia
    include("plotting_1d.jl")
```

For the three-dimensional examples, additional input files have to be obtained. This can be done in two ways.

1.  Download the additional files from ([example_data.zip](https://box.fu-berlin.de/s/AippyZbEZB64FmX); ~11 GB) and add them to the `examples` folder.
2.  Generate the input files using the electronic structure code __exciting__ (add link) using the provided input files. It should be noted that some of the computations require multiple thousands of CPU hours. The version of exciting used to generate the date for the article was in development at the time of submission and has git hash (add hash of commit). The features used are expected to be included in the next major release of the __exciting__ software.

The diamond and graphene examples can then be run using the commands
```julia
    include("benchmarks_3d.jl") # takes about  1200 seconds
    include("errors_3d.jl")     $ takes about 40000 seconds
    include("plotting_3d.jl")
```
