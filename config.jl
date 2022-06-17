## A configuration file to store important constants and other settings.

## Shorthands for the most frequently used types.
const AFloat = AbstractFloat;
const AVec = AbstractVector;
const AMat = AbstractMatrix;
const AArr = AbstractArray;
const AString = AbstractString;

## Some constants for frequently used paths.
const PROJECT_ROOT = @__DIR__;
const SRC_PATH = joinpath(PROJECT_ROOT, "src");
const JULIA_SRC_PATH = joinpath(SRC_PATH, "julia");
const LIB_PATH = joinpath(JULIA_SRC_PATH, "lib");
const SCRIPTS_PATH = joinpath(JULIA_SRC_PATH, "scripts");
const DATA_PATH = joinpath(PROJECT_ROOT, "data");
const SIMEXP_PATH = joinpath(PROJECT_ROOT, "output", "simulation-experiments");
const JLD2_PATH = joinpath(DATA_PATH, "jld2");

## Activate environment for this project.
import Pkg
Pkg.activate(PROJECT_ROOT, io = Base.DevNull()); # DevNull to activate silently.

## Modify LOAD_PATH such that packages are loaded only from this
# environment and the standard library.
empty!(LOAD_PATH);
push!(LOAD_PATH, "@"); # This environment.
push!(LOAD_PATH, "@stdlib"); # Standard library.

nothing;
