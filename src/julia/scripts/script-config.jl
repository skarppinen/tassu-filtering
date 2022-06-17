include("../../../config.jl");
import ArgParse

## Global object keeping all configurations for scripts.
# This object is read in script files.
ARGUMENT_CONFIG = Dict();

let
    s = ArgParse.ArgParseSettings();
    ArgParse.@add_arg_table! s begin
        "--npar", "-p"
        help = "the amount of particles"
        default = 64
        arg_type = Int

        "--nreps", "-M"
        help = "number of times to repeat the simulations"
        default = 50
        arg_type = Int

        "--intensity-model", "-m"
        help = string("which intensity model to use. ",
                      "1 = intensity varies wrt time, ",
                      "2 = intensity varies wrt time and smoothly in space, ",
                      "3 = 2 + Corine added")
         default = 3
        arg_type = Int

        "--lambda-bd"
        help = string("birth and death intensity");
        default = 0.0015
        arg_type = Float64

        "--lambda-b0"
        help = string("baseline birth intensity");
        default = 0.0
        arg_type = Float64

        "--lambda-clutter"
        help = string("intensity of clutter observations");
        default = 0.0
        arg_type = Float64

        "--lambda-obs"
        help = string("scaling of observation intensity");
        default = 1.0
        arg_type = Float64

        "--contour-prob"
            help = "the contour probability for selecting sigmaobs"
            arg_type = Float64
            default = 0.95
        "--diameter"
            help = "the assumed diameter of circle (in km) enclosed by `contour-prob`, used to set sigmaobs."
            arg_type = Float64
            default = 65.48545

        "--max-n-packs"
        help = string("maximum number of packs at any time during filtering.");
        default = 100
        arg_type = Int

        "--outfolder", "-o"
        help = "folder where to save data (relative from where script invoked)"
        arg_type = String

        "--verbose", "-v"
        help = "display messages about progress?"
        action = :store_true

        "--jobid", "-j"
        help = string("an integer used in naming to avoid naming clashes in parallel jobs.",
                      " if not set, a random name for output is used.");
        arg_type = Int
end
    global ARGUMENT_CONFIG["npar-check"] = s;
end

let
    s = ArgParse.ArgParseSettings();
    ArgParse.@add_arg_table! s begin
        "--npar", "-p"
            help = "the amount of particles"
            default = 64
            arg_type = Int
        "--nreps", "-M"
            help = "number of times to repeat the simulations"
            default = 20
            arg_type = Int
        "--intensity-model", "-m"
            help = string("which intensity model to use. ",
                          "1 = intensity varies wrt time, ",
                          "2 = intensity varies wrt time and smoothly in space, ",
                          "3 = 2 + Corine added")
            default = 1
            arg_type = Int

        "--max-n-packs"
            help = string("maximum number of packs at any time during filtering.");
            default = 100
            arg_type = Int

        "--lambda-bd"
            help = string("birth and death intensity");
            default = 0.0015
            arg_type = Float64

        "--lambda-b0"
            help = string("baseline birth intensity");
            default = 0.0
            arg_type = Float64

        "--lambda-clutter"
            help = string("intensity of clutter observations");
            default = 0.0
            arg_type = Float64

        "--lambda-obs"
            help = string("scaling of observation intensity");
            default = 1.0
            arg_type = Float64

        "--contour-prob"
            help = "the contour probability for selecting sigmaobs"
            arg_type = Float64
            default = 0.95
        "--diameter"
            help = "the assumed diameter of circle (in km) enclosed by `contour-prob`, used to set sigmaobs."
            arg_type = Float64
            default = 65.48545
        "--outfolder", "-o"
            help = string("folder where to save data (relative from where script invoked)")
            arg_type = String
        "--verbose", "-v"
            help = "display messages about progress?"
            action = :store_true
        "--jobid", "-j"
            help = string("an integer used in naming to avoid naming clashes in parallel jobs.",
                          " if not set, a random name for output is used.");
            arg_type = Int
    end
    global ARGUMENT_CONFIG["sim-w-covariates"] = s;
end

let
    s = ArgParse.ArgParseSettings();
    ArgParse.@add_arg_table! s begin
    "--npar", "-p"
        help = "the amount of particles"
        default = 64
        arg_type = Int
    "--intensity-model", "-m"
        help = string("which intensity model to use. ",
                      "1 = intensity varies wrt time, ",
                      "2 = intensity varies wrt time and smoothly in space, ",
                      "3 = 2 + Corine added")
        default = 3
        arg_type = Int
    "--lambda-bd"
        help = string("birth and death intensity");
        default = 0.0015
        arg_type = Float64
    "--lambda-b0"
        help = string("baseline birth intensity");
        default = 0.0
        arg_type = Float64
    "--lambda-clutter"
        help = string("intensity of clutter observations");
        default = 0.0
        arg_type = Float64
    "--lambda-obs"
        help = string("scaling of observation intensity");
        default = 1.0
        arg_type = Float64
    "--gating", "-g"
        help = string("if true, reduce computation by disregarding practically ",
                      "impossible observation associations. experimental feature.");
        action = :store_true
    "--contour-prob"
        help = "the contour probability for selecting sigmaobs"
        arg_type = Float64
        default = 0.95
    "--diameter"
        help = "the assumed diameter of circle (in km) enclosed by `contour-prob`, used to set sigmaobs."
        arg_type = Float64
        default = 65.48545
    "--max-n-packs"
        help = string("the maximal number of packs in any particle during filtering.")
        default = 100
        arg_type = Int
    "--use-known-locations"
        help = string("if set, initial distribution uses known territory locations from spring 2019.",
                      "otherwise initial distribution is a (30, 70)-truncated NegBin with mean 45.");
        action = :store_true
    "--randomize"
        help = string("if set, randomize the filtering seed. if not set, a hardcoded seed is used.",
                      " note that the discretiser seed is always hardcoded.");
        action = :store_true
    "--jobid"
        help = string("used in naming if `randomize` = true to avoid clashes.");
        default = 1
        arg_type = Int
    "--outfolder", "-o"
        help = string("folder where to save data (relative from where script invoked)")
        arg_type = String
    "--verbose", "-v"
        help = "display messages about progress?"
        action = :store_true
    end
    global ARGUMENT_CONFIG["filter-tassu"] = s;
end

let
    s = ArgParse.ArgParseSettings();
    ArgParse.@add_arg_table! s begin
        "--npar", "-p"
            help = "the amount of particles"
            default = 64
            arg_type = Int
        "--nreps", "-M"
            help = "number of times to repeat the simulations"
            default = 10
            arg_type = Int
        "--N-init", "-N"
            help = "the mean initial number of packs"
            default = 20
            arg_type = Int
        "--lambda-birth-death", "-l"
            help = "value for lambda birth and death"
            arg_type = Float64
            default = 0.0025
        "--lambda-birth0"
            help = "baseline birth intensity"
            arg_type = Float64
            default = 0.0
        "--lambda-obs"
            help = "scaling for observation intensity"
            arg_type = Float64
            default = 1.0
        "--lambda-clutter"
            help = "intensity of clutter observations"
            arg_type = Float64
            default = 0.0
        "--sigmaobs"
            help = "pack size parameter"
            arg_type = Float64
            default = 5.0
        "--outfolder", "-o"
            help = string("folder where to save data (relative from where script invoked)")
            arg_type = String
        "--verbose", "-v"
            help = "display messages about progress?"
            action = :store_true
        "--jobid", "-j"
            help = string("an integer used in naming to avoid naming clashes in parallel jobs.",
                          " if not set, a random name for output is used.");
            arg_type = Int
    end
    global ARGUMENT_CONFIG["bias-check"] = s;
end
