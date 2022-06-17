# Research code for the paper "Identifying territories using citizen science data: an application to the Finnish wolf population"

_Karppinen, Rajala, Mäntyniemi, Kojola, Vihola, "Identifying territories using presence-only citizen science data: an application to the Finnish wolf population", under revision for Ecological Modelling (2022)_

This repository contains code used to produce the results in Sections 3.2 - 3.4.
Note that the Tassu data is not public, so the code may only be run for experiments that do not directly use
the Tassu data. 

## Installation 

This repository features both Julia (version 1.5.4) and R (version 4.2.0) code. Start by cloning the repository.
Then follow the instructions below for Julia and R, respectively.

### Julia
Install Julia 1.5.4 and

1. Start the Julia REPL at the top folder in the directory hierarchy.
2. Run the following code in the REPL: 
```
import Pkg
Pkg.activate(".") # Activates environment.
Pkg.instantiate() # Downloads and installs required packages (based on Project.toml and Manifest.toml).
```

### R
Install R 4.2.0 and run the following code: 
```
install.packages(c("rnaturalearth", "ggspatial", "sf", "sp", "ggplot2", "dplyr",
                   "raster", "patchwork", "here", "leaflet", "webshot", "mapview",
                   "htmlwidgets", "RColorBrewer"))
```

## Documentation of directory hierarchy and most important files

### src/bash

Contains job lists that may be run in order to produce (the equivalent of) the results shown in the
paper. The files also document what parameters have been used to run the scripts in "src/julia/scripts". (see below)
On Linux, each joblist may be run by calling `bash *joblistname*` (or equivalent) on the command line.

Running all jobs in the joblists will take a long time on a single machine and using a cluster
is advised. 
The script "download-simulation-data.jl" (see below) may be used instead to download the simulation data.

### src/julia/lib

Contains source code for the sequential Monte Carlo algorithm, simulating observations etc. 
The main file is "wolftrack-filter.jl" which contains the inference algorithm.

### src/julia/scripts

Contains code mainly for running the experiments in the paper.
To run a script, call `julia *scriptname*` on the command line, followed by any arguments the script takes.

* Files starting with "run-" do the experiments (with arguments that can be set, see also below).
Call `julia *scriptname* --help` on the command line to see the documentation for each script.

* "script-config.jl" configures the scripts running the experiments. This script does not produce any output. 

* "download-simulation-data.jl" downloads simulation data to the folder "output/simulation-experiments"
that is created when this script is run. 

* "plot.jl" reproduces the main plots in the Results based on the simulation data. The plots
are produced to "output/plots" that is created when this script is run. The simulation
data must be available in "output/simulation-experiments" so that this script may run.

* "print-n-estimates.jl" reproduces the statistics shown in Table 1 based on files in "data" (see below).

### src/R

Contains R source files mainly used for making some of the Figures in the Results. 
All plotting scripts are called by "src/julia/scripts/plot.jl" (see above).
Therefore, these scripts need not be called separately.

* "plot-phds.R" produces Figure 6.
* "plot-study-area-and-continental-map.R" produces Figure 2 excluding the plot with Tassu data (Dataset C).
* "r-helpers.R" and "r-plot-helpers.R" provide utilities used in making the figures.

### data

Contains RDS (R datasets) in "data/rds" and JLD2 (Julia datasets) in "data/jld2".
These data are used by the scripts in "src/julia/scripts".

* "n-est*": Distribution of the number of territories estimated by the filter (spring 2020) for models 1-4.

* "official-territory-count-estimates-2020": Territory count estimates by Luke (2020).

* "pack-locations-simple-early-2019", "pack-locations-geometry-early-2019": Known territory locations from assessment of spring by Luke (2019).

* "phd*": The (truncated) probability hypothesis densities for models 1-4.

* "spatpred-master": Results of intensity analysis, spatial effect (model 3 in data corresponds to full Poisson regression, model 1 corresponds to model with no spatial covariates in Poisson regression)

* "study-area-polygon-wgs84": Polygon of the study area in WGS84.

* "territory-polygons-2020": Polygons of the wolf territories found by Luke (2020).

* "timepred-linreg": Results of intensity analysis, temporal effect.
