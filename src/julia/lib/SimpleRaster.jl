## Simplistic implementation of a basic raster object.
include("../../../config.jl");

import ArchGDAL
using StaticArrays
import GDAL
const AG = ArchGDAL;
using DataFrames
include(joinpath(LIB_PATH, "type-aliases.jl"));
include(joinpath(LIB_PATH, "bbox.jl"));

"""
An information object for a SimpleRaster.
"""
struct RasterInformation
    # Bounding box.
    bbox::BoundingBox{Float64, 2, 4}

    n_x::Int # Number of pixels in x direction.
    n_y::Int # Number of pixels in y direction.
    ps_x::Float64 # Pixel width.
    ps_y::Float64 # Pixel height.

    function RasterInformation(bbox::BoundingBox{T, 2}, n_x::Integer,
                               n_y::Integer) where {T <: Real}
        @assert n_x >= 1 "`n_x` must be greater than 1";
        @assert n_y >= 1 "`n_y` must be greater than 1";
        ps_x = (bbox[2] - bbox[1]) / n_x;
        ps_y = (bbox[4] - bbox[3]) / n_y;
        new(bbox, n_x, n_y, ps_x, ps_y);
    end
end

"""
Fetch information about a raster dataset.
Legend for the geotransform:
gt[1] = "x coordinate of the upper left corner of the upper left pixel"
gt[2] = "pixel width (x direction)"
gt[3] = "skew coefficient for x"
gt[4] = "y coordinate of the upper left corner of the upper left pixel"
gt[5] = "skew coefficient for y"
gt[6] = "pixel height (y direction)"

Note that the pixel widths can well be negative.
See:
https://gdal.org/doxygen/classGDALPamDataset.html (GetGeoTransform)
https://gis.stackexchange.com/questions/314654/gdal-getgeotransform-documentation-is-there-an-oversight-or-what-am-i-misund
"""
function RasterInformation(d, bi::Int = 1)
    band = AG.getband(d, bi);
    gt = AG.getgeotransform(d);

    @assert gt[2] > 0.0 "SimpleRaster requires positive pixel width in raster";
    @assert gt[6] < 0.0 "SimpleRaster requires negative pixel height in raster.";
    @assert isapprox(gt[3], 0.0) "SimpleRaster does not support skew parameters, see documentation for RasterInformation.";
    @assert isapprox(gt[5], 0.0) "SimpleRaster does not support skew parameters, see documentation for RasterInformation.";

    xmin = gt[1]; ymax = gt[4];
    n_x = convert(Int, AG.width(band));
    n_y = convert(Int, AG.height(band));
    xmax = xmin + gt[2] * n_x;
    ymin = ymax + gt[6] * n_y;
    ps_x = (xmax - xmin) / n_x;
    ps_y = (ymax - ymin) / n_y;

    bbox = BoundingBox((xmin, xmax, ymin, ymax));
    RasterInformation(bbox, n_x, n_y);
end

"""
A simple raster object that follows the computer graphics convention that
the top left corner is the origin and x increases with increasing column
and y with increasing row.
"""
struct SimpleRaster{R}
    r::Matrix{R} # The raster data.
    mis::R # Value representing missing values in the raster.
    info::RasterInformation # Information about raster, bounding box.. etc.
    function SimpleRaster(r::AMat{R}, mis::R, info::RasterInformation) where R
        @assert size(r, 1) == info.n_y "raster dimensions do not match with raster information";
        @assert size(r, 2) == info.n_x "raster dimensions do not match with raster information"
        new{R}(r, mis, info);
    end
end

"""
Transform a SimpleRaster to a new SimpleRaster with the raster values transformed
with a lookup table.
"""
function transform(sr::SimpleRaster{R},
                   lookup::Dict{R, <: AFloat}) where R
    new_mis = lookup[sr.mis];
    new_r = map(x -> lookup[x], sr.r);
    SimpleRaster(new_r, new_mis, sr.info);
end

"""
Load an ArchGDAL band as a SimpleRaster object.

Arguments:
`d`: The ArchGDAL dataset.
`bi`: The index of the band to load.
"""
function SimpleRaster(d::ArchGDAL.Dataset, bi::Int = 1)
    # Get raster information.
    info = RasterInformation(d, bi);

    # Load band from data.
    r = AG.read(d, bi);
    band = AG.getband(d, bi);
    dtype = AG.getdatatype(band);
    mis = convert(dtype, AG.getnodatavalue(band));

    SimpleRaster{dtype, typeof(r)}(transpose(r)[info.n_y:-1:1, :], mis, info);
end

"""
Load a subset specified by a bounding box from an ArchGDAL band as a SimpleRaster object.

Arguments:
`d`: The ArchGDAL dataset.
`bbox`: The bounding box covering the area to load.
`bi`: The index of the band to load.
"""
function SimpleRaster(d::ArchGDAL.Dataset, bbox::BoundingBox, bi::Int = 1)
    # Get information about the full raster.
    # Here we also check that the orientation of the data is adequate
    # for SimpleRaster. Else an error will be thrown here.
    info = RasterInformation(d, bi);

    # Check bounding box given as a parameter, must be inside bounding box of
    # full raster.
    bbox_full = info.bbox; # Bounding box of the full raster.
    @assert bbox.xmin >= bbox_full.xmin "xmin in `bbox` must be greater than the minimum in the bbox of the data";
    @assert bbox.ymin >= bbox_full.ymin "ymin in `bbox` must be greater than the minimum in the bbox of the data";
    @assert bbox.ymax <= bbox_full.ymax "ymax in `bbox` must be lesser than the maximum in the bbox of the data";
    @assert bbox.xmax <= bbox_full.xmax "xmax in `bbox` must be lesser than the maximum in the bbox of the data";

    # Compute indices in the full raster which correspond to the bounding box.
    i_xmin = ceil(Int, (bbox.xmin - bbox_full.xmin) / info.ps_x);
    i_xmin = i_xmin == 0 ? i_xmin + 1 : i_xmin;
    i_ymin = ceil(Int, (bbox.ymin - bbox_full.ymin) / info.ps_y);
    i_ymin = i_ymin == 0 ? i_ymin + 1 : i_ymin;
    i_xmax = ceil(Int, (bbox.xmax - bbox_full.xmin) / info.ps_x);
    i_ymax = ceil(Int, (bbox.ymax - bbox_full.ymin) / info.ps_y);

    # Compute the bbox that "snaps" to the grid specified by the raster pixels.
    # The bbox given as a parameter is likely not exactly at pixel boundaries.
    xmin = bbox_full.xmin + (i_xmin - 1) * info.ps_x;
    ymin = bbox_full.ymin + (i_ymin - 1) * info.ps_y;
    xmax = bbox_full.xmin + i_xmax * info.ps_x;
    ymax = bbox_full.ymin + i_ymax * info.ps_y;
    bbox_new = BoundingBox(xmin, xmax, ymin, ymax);

    # Build the RasterInformation object.
    n_x = i_xmax - i_xmin + 1;
    n_y = i_ymax - i_ymin + 1;
    info_new = RasterInformation(bbox_new, n_x, n_y);

    # Load raster data. Here the small y values in world coordinates are at
    # the bottom of the raster. (So a north up picture is correctly
    # orientated). Hence the flipped indexing to the pixels values.
    band = AG.getband(d, bi);
    dtype = AG.getdatatype(band);
    mis = convert(dtype, AG.getnodatavalue(band));
    r = AG.read(band, (info.n_y - i_ymax + 1):(info.n_y - i_ymin + 1), i_xmin:i_xmax);

    # Create SimpleRaster object.
    # The transpose is because for some reason AG.read transposes its output.
    # Here we also flip the y axis, so small y values in world coordinates
    # are at the top of the raster, as is the convention in computer graphics.
    SimpleRaster{dtype}(transpose(r)[n_y:-1:1, :], mis, info_new);
end

"""
Extract a value from a SimpleRaster{R} object.

Arguments:
`sr`: A SimpleRaster of type SimpleRaster{R}
`x`: The x coordinate of the point (of type Real).
`y`: The y coordinate of the point (of type Real).

Example:
extract(sr, 1000.823, 321.4)

Note that here we assume that each tile is a rectangle
(xmin, xmax] × (ymin, ymax], i.e each tile contains the upper boundaries,
but not the lower.
"""
function extract(sr::SimpleRaster{R}, x::Real, y::Real) where R
    row, col = get_pixel_coordinates(sr, x, y);
    if row == -1 || col == -1
        return sr.mis;
    end
    sr.r[row, col];
end

"""
Alternative call to `extract`.
"""
@inline function (sr::SimpleRaster)(x::AVec{<: Real})
    @inbounds extract(sr, x[1], x[2]);
end

"""
Alternative call to `extract`.
"""
@inline function (sr::SimpleRaster)(x::Tuple{<: Real, <: Real})
    @inbounds extract(sr, x[1], x[2]);
end

"""
Compute the integral over the raster when the raster is thought of as a
pixelwise constant function from R^2 -> R.
"""
function integral(sr::SimpleRaster{R}) where {R <: Real}
    I = Float64(0.0);
    area = sr.info.ps_x * sr.info.ps_y;
    for v in sr.r
        if v != sr.mis
            I += area * v;
        end
    end
    I;
end

"""
Return the pixel value of the pixel where the point (x, y) falls.
If the point (x, y) is outside the bounding box of the raster, (-1, -1) is
returned.
"""
function get_pixel_coordinates(sr::SimpleRaster,
                               x::Real, y::Real)
    get_pixel_coordinates(sr.info, x, y);
end

"""
Return the pixel value of the pixel where the point (x, y) falls.
If the point (x, y) is outside the bounding box of the raster, (-1, -1) is
returned.
Arguments:
* `bbox`: The bounding box of the raster.
* `ps`: A tuple or some indexable contained with the pixel sizes in x and y
directions. The x pixel size should be in the first slot, and the y pixel size
in the second.
* `x, y`: The query point.
"""
function get_pixel_coordinates(info::RasterInformation,
                               x::Real, y::Real)
    @inbounds xmin = info.bbox[1]; xmax = info.bbox[2];
    @inbounds ymin = info.bbox[3]; ymax = info.bbox[4];
    @inbounds psx = info.ps_x;
    @inbounds psy = info.ps_y;

    col = ceil(Int, (x - xmin) / psx);
    row = ceil(Int, (y - ymin) / psy);
    if x > xmax || x <= xmin
        col = -1;
    end
    if y > ymax || y <= ymin
       row = -1;
    end
    (row, col);
end

"""
Find the midpoint of the pixel on row `yi` and column `xi`.
"""
function pixel_midpoint(info::RasterInformation, xi::Int, yi::Int)
    bbox = info.bbox;
    x = bbox[1] + (xi - 0.5) * info.ps_x;
    y = bbox[3] + (yi - 0.5) * info.ps_y;
    @assert x <= bbox[2] "`xi` out of bounds!";
    @assert y <= bbox[4] "`yi` out of bounds!";
    (x, y);
end


"""
Return a BitArray where the nonmissing pixels of the raster are marked with
ones. The argument `rinfo` contains the data about the desired output mask.
(can have different bounding box and resolution)
"""
function nonmissingmask(sr::SimpleRaster, rinfo::RasterInformation = sr.info)
    s = (rinfo.n_y, rinfo.n_x);
    out = BitArray(undef, s[1], s[2]); out .= false;
    for j in 1:size(out, 2)
        for i in 1:size(out, 1)
            # Find coordinates of pixel x = j, y = i.
            mp = pixel_midpoint(rinfo, j, i);

            # Find value in original raster.
            sr(mp) != sr.mis && (out[i, j] = true);
        end
    end
    out;
end
