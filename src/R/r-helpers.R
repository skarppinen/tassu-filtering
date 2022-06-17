library(raster)
library(leaflet)
library(ggplot2)
library(webshot)
library(htmlwidgets)
library(mapview)
library(patchwork)
library(sf)
library(sp)
library(here)

proj_ETRSTM35 <- "+proj=utm +zone=35 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
proj_WGS84 <- "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"

last <- function(x) {
  stopifnot(length(x) >= 1)
  x[length(x)]
}

# Make a basemap with good view over the region of interest.
leaflet_basemap <- function() {
  leaflet() %>% 
    setView(lng = 26.193022, lat = 62.979954, zoom = 6) %>% 
    addTiles()
}

last_phd <- function(phd) {
  phd[, , last(dim(phd))]
}

save_leaflet_map_plot <- function(map, outfile = "", width_px = 506, height_px = 614, zoom = 5) {
  
  if (outfile == "") {
    outfile <- file.path(paste0(tempfile(), ".png"))
  }
  mapshot(map, file = outfile, vwidth = width_px, vheight = height_px, 
          #cliprect = "viewport", 
          delay = 0.1, zoom = zoom) # Zoom increases resolution.
  
  outfile
}

plot_map_with_phd_overlay <- function(r, xlim, ylim, misval = 0.0, ready_for_leaflet = FALSE,
                                      expert_terr_year = -1, color_scale = c(NA, NA)) {
  proj_ETRSTM35 <- "+proj=utm +zone=35 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
  proj_EPSG_3857 <- "+proj=merc +lon_0=0 +k=1 +x_0=0 +y_0=0 +a=6378137 +b=6378137 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
  
  # Build initial map.
  basemap <- leaflet_basemap()
  
  if (!ready_for_leaflet) {
    # Make raster.
    yrev <- nrow(r):1
    r <- raster(r[yrev, ], crs = CRS(proj_ETRSTM35),
                xmn = xlim[1], xmx = xlim[2],
                ymn = ylim[1], ymx = ylim[2]) 
    r[r == misval] <- NA
    
    # Project raster to CRS used by Leaflet.
    r <- projectRasterForLeaflet(r, method = "bilinear") #crs = CRS(proj_EPSG_3857))  
  }
  
  color_scale_min <- color_scale[1]
  color_scale_max <- color_scale[2]
  
  # Set boundary values for color scale.
  if (is.na(color_scale_min)) {
    color_scale_min <- minValue(r)
  }
  if (is.na(color_scale_max)) {
    color_scale_max <- maxValue(r)
  }
  
  # Set color scheme.
  domain <- c(color_scale_min, color_scale_max)
  pal <- colorNumeric(c("#00000000", "#FF0000FF"), domain, na.color = "#00000000", alpha = TRUE)
  
  # Map with raster overlay.
  out <- basemap %>% 
           addRasterImage(r, opacity = 1.0, color = pal, project = FALSE)
  
  # Add expert territories if requested. -1 means not requested.
  if (expert_terr_year >= 0) {
    filepath <- file.path(here(), "data", "rds", paste0("territory-polygons-", expert_terr_year, ".rds"))
    if (!file.exists(filepath)) {
      stop(paste0("Attempted to add expert territories, could not find RDS file at ", filepath))
    }
    # A bbox used to drop territories from the north.
    north_crop_bbox <- st_bbox(c(xmin = 0, xmax = 100000000, 
                                 ymin = 0, ymax = max(72e5 + 30000)), 
                               crs = CRS(proj_ETRSTM35))
    # Load expert territories as sf object.
    expert_poly_sf <- st_as_sf(spTransform(readRDS(filepath), CRS(proj_ETRSTM35)))
    
    # Crop territories in the north away.
    crop_poly <- st_crop(expert_poly_sf$geometry, north_crop_bbox)
    
    # Add to plot.
    out <- out %>% 
      addPolylines(data = st_transform(crop_poly, CRS(proj_WGS84)),
                   weight = 4, color = "black")
  }
  out
}

image_concat <- function(img1, img2, outfile = "", mode = "+") {
  if (outfile == "") {
    outfile = file.path(paste0(tempfile(), ".png"))
  }
  cmd <- paste0(mode, "append")
  system2("convert", args = c(img1, img2, cmd, outfile))
  outfile
} 

# Make pngs of maps.
maps_to_png <- function(paths_to_rds, outfile = "", wscale = 506, hscale = 614, zoom = 5,
                        expert_terr_year = 2020, misval = 0.0, cutoff = NA) {
  if (outfile == "") {
    outfile <- file.path(paste0(tempfile(), ".png"))
  }
  paths <- vector("character", length(paths_to_rds))
  
  # Load PHD data.
  raster_list <- lapply(paths_to_rds, function(path) {
    inp <- readRDS(path)
    mat <- last_phd(inp[["phd"]])
    yrev <- nrow(mat):1
    xlim <- inp[["phd-xlim"]]; ylim <- inp[["phd-ylim"]]
    r <- raster(mat[yrev, ], crs = CRS(proj_ETRSTM35),
                xmn = xlim[1], xmx = xlim[2],
                ymn = ylim[1], ymx = ylim[2]) 
    r[r == misval] <- NA
    if (!is.na(cutoff)) {
      r[r > cutoff] <- cutoff
    }
    
    rproj <- projectRasterForLeaflet(r, method = "bilinear")
    print(paste0("min value is ", minValue(rproj)))
    print(paste0("max value is ", maxValue(rproj)))
    
    list(rproj = rproj,
         xlim = xlim, ylim = ylim)
  })
  
  # Find out maximum of all PHD values across maps.
  color_scale_min <- min(vapply(raster_list, function(x) minValue(x$rproj), numeric(1)))
  color_scale_max <- max(vapply(raster_list, function(x) maxValue(x$rproj), numeric(1)))
  color_scale <- c(color_scale_min, color_scale_max) 
  print(paste0("global color scale is ", color_scale[1], " - ", color_scale[2]))
  
  for (i in seq_along(raster_list)) {
    rproj <- raster_list[[i]]$rproj
    xlim <- raster_list[[i]]$xlim
    ylim <- raster_list[[i]]$ylim
    map <- plot_map_with_phd_overlay(rproj, xlim, ylim, 
                                     expert_terr_year = expert_terr_year,
                                     color_scale = color_scale, ready_for_leaflet = TRUE)
    paths[i] <- save_leaflet_map_plot(map, width_px = wscale, height_px = hscale, 
                                      zoom = zoom)
  } 
  paths
} 

map_quad_to_png <- function(paths, outfile = "") {
  stopifnot(length(paths) == 4)
  if (outfile == "") {
    outfile <- file.path(paste0(tempfile(), ".png"))
  }
  
  left <- image_concat(paths[1], paths[2], mode = "+")
  right <- image_concat(paths[3], paths[4], mode = "+")
  path_to_png <- image_concat(left, right, outfile = outfile, mode = "+")
  path_to_png
}

maps_to_2by5 <- function(paths, outfile = "") {
  stopifnot(length(paths) == 10)
  if (outfile == "") {
    outfile <- file.path(paste0(tempfile(), ".png"))
  }
  
  top <- paths[1]
  for (i in 2:5) {
    top <- image_concat(top, paths[i], mode = "+")
  }
  bottom <- paths[6]
  for (i in 7:10) {
    bottom <- image_concat(bottom, paths[i], mode = "+")
  }
  image_concat(top, bottom, outfile = outfile, mode = "-")
}

