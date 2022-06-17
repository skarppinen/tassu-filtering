include(joinpath(LIB_PATH, "wolfpack-types.jl"));
include(joinpath(LIB_PATH, "observation-simulator.jl"));
include(joinpath(LIB_PATH, "spatial-operations.jl"));
using Random
using DataFrames, CategoricalArrays

"""
Read a CSV file saved in R with the call
`write.csv(file = *path_to_file*, row.names = FALSE)`.
Might not work for some special cases but handles missing values.
"""
function read_R_csv(filepath::AbstractString)
    d = CSV.read(filepath, DataFrame; missingstring = "NA");
    d;
end

"""
Read observations from a CSV and return a data object that can be plugged into the
wolf pack filter. The input CSV must be structured such that it contains exactly
three columns with the column names `time`, `x` and `y`.
The time column contains the time of the observation, and the x and y columns
contain the 2D coordinates of the observation. If loading a CSV saved in R,
save the CSV with `write.csv`, use row.names = FALSE and leave the remaining
options to their defaults.

Arguments:
`filepath`: The path to the input CSV.

Optional keyword arguments:
`preprocess`: A boolean value stating whether the data preprocessing step should
be applied to the data. Default value is true.
"""
function read_wolfpack_observations(filepath::AbstractString;
                                    preprocess::Bool = true)
   d = read_R_csv(filepath);
   format_observations(d; preprocess = preprocess);
end

"""
Format observations in a DataFrame such that they can be plugged into the
wolf pack filter. The return value of the function is a vector of observations.
The input DataFrame must be structured such that it contains exactly three columns
with the column names `time`, `x` and `y`. The time column contains the time of
the observation, and the x and y columns contain the 2D coordinates of the
observation.

Arguments:
`d`: An input DataFrame.

Optional keyword arguments:
`preprocess`: A boolean value stating whether the data preprocessing step should
be applied to the data. Default value is true.
"""
function format_observations(d::DataFrame;
                             preprocess::Bool = true)
   nm = Symbol.(names(d));
   @assert length(nm) == 3 "the input data must contain exactly 3 columns.";
   @assert :time in nm string("the input data must contain a column named ", "`time`.");
   @assert :x in nm string("the input data must contain a column named ", "`x`.");
   @assert :y in nm string("the input data must contain a column named ", "`y`.");
   for typ in [eltype(c) for c in eachcol(d)]
      @assert typ <: Real "all columns of the input data must have a real valued type. " *
                          "Check that the type of some column is not Union{.., Missing}!";
   end
   time = d[:, :time];
   @assert issorted(time) "the `time` column in the input data must be increasing.";
   dt = diff(vcat([0.0], time));
   x = d[:, :x]; y = d[:, :y];
   if !preprocess
      return map(time, dt, x, y) do t, dt, x, y
         WolfpackObs(SVector{2, Float64}(x, y), t, dt);
      end;
   else
      return discretiser(time, map((x, y) -> [x, y], x, y), time[end]);
   end
end


"""
Convert a vector of 2D WolfpackObs' to a DataFrame.
"""
function DataFrame(v::AVec{<: WolfpackObs{2}})
    time = getfield.(v, :time);
    x = map(x -> x.location[1], v);
    y = map(x -> x.location[2], v);
    DataFrame(time = time, x = x, y = y);
end

"""
Produce a new dataset that contains the original data + the timepoints in `tp`.
Observations of type WolfpackMisObs are added such that the output contains the
union of the timepoints in `data` and `tp`.
Values in `tp` should be within the range of the data times (not checked).
"""
function add_timepoints(data::AVec{<: Observation{N}},
                        tp::AVec{<: Real}) where N
   orig_times = getfield.(data, :time);
   new_times = sort!(union(orig_times, tp)); # Times in new data.
   new_dt = diff(vcat([0.0], new_times)); # dt in new data.
   real_obs_i = map(new_times) do time
      i = findfirst(x -> isapprox(x, time), orig_times);
      if i == nothing || typeof(data[i]) <: WolfpackMisObs
          return -1;
      end
      i;
   end

   # Build new dataset.
   out = Vector{Observation{N}}(undef, 0);
   sizehint!(out, length(data) + length(tp));
   for i in 1:length(new_times)
      obs_i = real_obs_i[i];
      time = new_times[i];
      dt = new_dt[i];
      if obs_i != -1
         push!(out, WolfpackObs(copy(data[obs_i].location), time, dt));
      else
         push!(out, WolfpackMisObs(time, dt));
      end
   end
   out;
end

function read_jld2_dataframe(filename::AString)
    jldopen(joinpath(JLD2_PATH, filename * ".jld2"), "r") do file
        file["out"];
    end
end

function nextodd(x::Integer)
    isodd(x) ? x : x + 1;
end

function get_covariate_data(time_covariate_file::AString;
                            model::Integer,
                            σ_km::Real = 0.0,
                            l = nextodd(ceil(Int, 2 * 2 * σ_km)),
                            bbox_padding_km::Integer = 0)


    # Load raw time covariate data.
    t, cm = get_raw_time_covariate_data(time_covariate_file; model = model,
                                                   include_month_length = true);

    # Load raw spatial covariate data.
    spatdf_all = read_jld2_dataframe("spatpred-master");
    spatdf = filter(r -> r[:model] == model, spatdf_all);

    # Settings for spatial covariate data.
    pixel_size = 1000.0;
    Vsize = pixel_size * pixel_size;
    mis = 0.0;
    xmin = minimum(spatdf[!, :x]) - pixel_size / 2.0 - bbox_padding_km * pixel_size;
    xmax = maximum(spatdf[!, :x]) + pixel_size / 2.0 + bbox_padding_km * pixel_size;
    ymin = minimum(spatdf[!, :y]) - pixel_size / 2.0 - bbox_padding_km * pixel_size;
    ymax = maximum(spatdf[!, :y]) + pixel_size / 2.0 + bbox_padding_km * pixel_size;
    bb = BoundingBox((xmin, xmax, ymin, ymax));
    n_x = convert(Int, (xmax - xmin) / pixel_size);
    n_y = convert(Int, (ymax - ymin) / pixel_size);

    # Apply Gaussian blur if \sigma_km is positive.
    if σ_km > 0.0
        # Do blurring in linear scale.
        spatdf[!, :lambda] .= exp.(spatdf[!, :log_lambda]);
        r = fill(mis, n_y, n_x);
        for i in 1:nrow(spatdf)
            x = spatdf[i, :x] + pixel_size / 2.0;
            y = spatdf[i, :y] + pixel_size / 2.0;
            xi = convert(Int, (x - xmin) / pixel_size);
            yi = convert(Int, (y - ymin) / pixel_size);
            r[yi, xi] = spatdf[i, :lambda];
        end
        # Do Gaussian blur.
        kernel = build_blur_kernel(σ_km, l, mis);
        r = mapwindow(kernel, r, (l, l), border = Fill(mis));

        # Compute blurred log-scale lambda.
        for i in 1:nrow(spatdf)
            x = spatdf[i, :x] + pixel_size / 2.0;
            y = spatdf[i, :y] + pixel_size / 2.0;
            xi = convert(Int, (x - xmin) / pixel_size);
            yi = convert(Int, (y - ymin) / pixel_size);
            spatdf[i, :log_lambda] = log(r[yi, xi]);
        end
    end

    # Build time covariate.
    linvV = log(inv(Vsize));
    K = -maximum(spatdf[!, :log_lambda]) - linvV;
    cm .= exp.(cm .- K);

    # Build spatial covariate.
    r = fill(mis, n_y, n_x);
    for i in 1:nrow(spatdf)
        x = spatdf[i, :x] + pixel_size / 2.0;
        y = spatdf[i, :y] + pixel_size / 2.0;
        xi = convert(Int, (x - xmin) / pixel_size);
        yi = convert(Int, (y - ymin) / pixel_size);
        r[yi, xi] = exp(linvV + spatdf[i, :log_lambda] + K);
    end
    info = RasterInformation(bb, n_x, n_y);
    sr = SimpleRaster(r, mis, info);
    PiecewiseConstant(t, cm), sr;
end

function get_raw_spatial_grid(model::Integer;
                              bbox_padding_km::Integer = 0)

    # Load raw spatial covariate data.
    spatdf_all = read_jld2_dataframe("spatpred-master");
    spatdf = filter(r -> r[:model] == model, spatdf_all);

    # Settings for spatial covariate data.
    pixel_size = 1000.0;
    Vsize = pixel_size * pixel_size;
    mis = 0.0;
    xmin = minimum(spatdf[!, :x]) - pixel_size / 2.0 - bbox_padding_km * pixel_size;
    xmax = maximum(spatdf[!, :x]) + pixel_size / 2.0 + bbox_padding_km * pixel_size;
    ymin = minimum(spatdf[!, :y]) - pixel_size / 2.0 - bbox_padding_km * pixel_size;
    ymax = maximum(spatdf[!, :y]) + pixel_size / 2.0 + bbox_padding_km * pixel_size;
    bb = BoundingBox((xmin, xmax, ymin, ymax));
    n_x = convert(Int, (xmax - xmin) / pixel_size);
    n_y = convert(Int, (ymax - ymin) / pixel_size);

    # Build spatial covariate.
    r = fill(mis, n_y, n_x);
    for i in 1:nrow(spatdf)
        x = spatdf[i, :x] + pixel_size / 2.0;
        y = spatdf[i, :y] + pixel_size / 2.0;
        xi = convert(Int, (x - xmin) / pixel_size);
        yi = convert(Int, (y - ymin) / pixel_size);
        r[yi, xi] = exp(spatdf[i, :log_lambda]);
    end
    info = RasterInformation(bb, n_x, n_y);
    sr = SimpleRaster(r, mis, info);
    sr;
end


function month_length_days(month::Int, year::Int)
    base_lengths = (31.0, 28.0, 31.0, 30.0, 31.0, 30.0,
                    31.0, 31.0, 30.0, 31.0, 30.0, 31.0);
    if month == 2 && Dates.isleapyear(year)
        return 29.0;
    end
    base_lengths[month];
end

function month_from_wolf_month(wolf_month::Integer)
    @assert 1 <= wolf_month <= 12 "`wolf_month` should be between 1 and 12.";
    (wolf_month - 1 + 3) % 12 + 1;
end

function get_year_and_month_from_wolf_time!(d::DataFrame; colnames::NTuple{2, Symbol} = (:vuosif, :kkf))
    months = month_from_wolf_month.(parse.(Int, string.(d[:, :wolf_month])));
    years = Vector{String}(undef, length(months));
    for i in eachindex(years)
        lwr = match(r"^[[:digit:]]{4}", string.(d[i, :wolf_year]));
        upr = match.(r"[[:digit:]]{4}$", string.(d[i, :wolf_year]));
        if lwr == upr == nothing
            years[i] = String(d[i, :wolf_year]);
        else
            years[i] = 4 <= months[i] <= 12 ? lwr.match : upr.match;
        end
    end
    for t in zip(colnames, (years, months))
        d[!, t[1]] .= t[2];
    end
    nothing;
end

function is_valid_wolf_year_string(wolf_year::AString)
    re = Regex(raw"^[[:digit:]]{4}-[[:digit:]]{4}$")
    isnothing(match(re, wolf_year)) && (return false;)
    return true;
end

function years_from_wolfyear_str(wolf_year::AString)
    startyear_m = match(r"^[[:digit:]]{4}", wolf_year);
    endyear_m = match(r"[[:digit:]]{4}$", wolf_year);
    if isnothing(startyear_m) || isnothing(endyear_m)
        throw(ArgumentError("input is not a valid wolf year string."));
    end
    startyear = parse(Int, startyear_m.match);
    endyear = parse(Int, endyear_m.match);
    if endyear != (startyear + 1)
        throw(ArgumentError("input is parseable but not having consecutive years."));
    end
    (startyear, endyear);
end


function get_raw_time_covariate_data(filename::AString; model::Int = 3,
                                 wolf_year::AString = "2019-2020",
                                 include_month_length::Bool = true)
    @assert model in (1, 2, 3) "`model` must be in (1, 2, 3)";
    wolf_year = replace(wolf_year, " " => ""); # Drop white space.

    timedf = read_jld2_dataframe(filename);
    timedf = filter(r -> r[:model] == model && r[:wolf_year] == wolf_year, timedf);
    @assert nrow(timedf) > 0 "something wrong. loaded time prediction data after filtering is empty!";


    # NOTE: Here assumed that loaded data has correct order, i.e first month is
    # April, second is May and so on i.e that the input data has data in terms
    # of "wolf years".
    y = timedf[:, :log_lambda]; # Values at each interval of the function.
    startyear, endyear = years_from_wolfyear_str(wolf_year);
    ml = vcat(month_length_days.(4:12, startyear),
              month_length_days.(1:3, endyear)); # Month lengths.
    t = cumsum(vcat([0.0], ml)); # Time knots for piecewise constant function.

    if include_month_length
        y .= y .- log.(ml);
    end
    t[end] = nextfloat(t[end]); # Ensure that there is a bit after last time point.
    t, y;
end

"""
Get time covariate data starting from the beginning of year `startyear` in
month `startmonth`.
Arguments:
`model`: 1, 2 or 3. Which model to load.
`interval_length`: How long is the interval (days) covariate data is requested for.
`read_wolf_year_data`: Set true if want to read a dataset that has time given
in "wolf years" and "wolf months". (default false)
`force_year_eff`: A string corresponding to a year to use in predicting, always.
If empty, then the "mean year effect" is used for predictions, but data values are used if
requested interval spans data (i.e for months that exist in data, no prediction needed).
`include_interval_length`: If true, the length of each month is included to
the computed intensity values.
"""
function get_time_covariate_data(startyear::Integer,
                                 startmonth::Integer;
                                 model::Int,
                                 nmonths::Int = 12,
                                 read_wolf_year_data::Bool = false,
                                 force_year_eff::String = "",
                                 include_interval_length::Bool = true)
    @assert model in (1, 2, 3) "`model` must be in (1, 2, 3)";
    @assert 1 <= startmonth <= 12 "`startmonth` must be between 1 and 12.";
    year_eff_forced = !isempty(force_year_eff);
    !year_eff_forced && (force_year_eff = "mean";);

    # Get prediction data.
    if read_wolf_year_data
        timedf = read_jld2_dataframe("timepred-master-wolf-years");
        get_year_and_month_from_wolf_time!(timedf);
    else
        timedf = read_jld2_dataframe("timepred-master");
    end
    timedf = filter(r -> r[:model] == model, timedf);

    cur_year = startyear;
    block_year_incr = startmonth == 1;
    cumulative = 0.0;
    t = [0.0]; # Time knots for piecewise constant function.
    y = Vector{Float64}(undef, 0); # Values at each interval of the function.

    # Walk monthly data until no time remains.
    for i in startmonth:(startmonth + nmonths - 1)
        cur_month = (i - 1) % 12 + 1; # Get month.

        # Handle year change. (basically increment year if not first iter and startmonth == 1)
        if cur_month == 1 && !block_year_incr
            cur_year += 1;
            block_year_incr = false;
        end

        # Get month length.
        ml = month_length_days(cur_month, cur_year);

        # Get lambda value from data.
        mλ = let
            # Find prediction from data (if there is one).
            pred_df = filter(r -> string(r[:vuosif]) == string(cur_year) &&
                                  string(r[:kkf]) == string(cur_month), timedf);

            v = if nrow(pred_df) == 0 || year_eff_forced
                # There is no prediction in data or using mean year effect is forced.
                pred = filter(r -> string(r[:vuosif]) == force_year_eff &&
                                   string(r[:kkf]) == string(cur_month), timedf);
                pred[1, :log_lambda];
            else
                # There is a prediction in data. (use it)
                pred_df[1, :log_lambda];
            end
            include_interval_length && (v += -log(ml));
            v;
        end
        push!(t, cumulative + ml);
        push!(y, mλ);
        cumulative += ml;
    end
    t[end] = nextfloat(t[end]); # Ensure that there is a bit after last time point.
    t, y;
end
