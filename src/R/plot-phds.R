library(here)
source(file.path(here(), "src/R/r-helpers.R"), local = TRUE)

files <- c("phd-npar16384-m3-locprior-prob0.95-diam65.48545-bd0.0015",
           "phd-npar16384-m3-locprior-prob0.95-diam65.48545-bd0.0015-clut0.475",
           "phd-npar16384-m1-locprior-prob0.95-diam65.48545-bd0.0015",
           "phd-npar16384-m1-locprior-prob0.95-diam65.48545-bd0.0015-clut0.475")
files <- file.path(here(), "data/rds", paste0(files, ".rds"))

# Compute cutoff.
o <- readRDS(files[1])

# NOTE: Cutoff multiplied by 1000 * 1000 (pixel size 1km * 1km) since values in PHD also are.
# (this scaling does not affect image)
ps <- o$`phd-psx`
cutoff <- dnorm(0.0, sd = o$`theta-sigmaobs`) ^ 2 * ps * ps

# Make plot.
png_files <- maps_to_png(files, cutoff = cutoff)
map_quad_to_png(png_files, outfile = file.path(here(), "output", "plots", "phd-1-4.png"))