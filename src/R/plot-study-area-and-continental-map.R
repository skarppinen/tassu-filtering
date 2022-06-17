library(rnaturalearth) # For getting map data.
library(ggspatial) # For adding scale and orientation.
library(sf)
library(ggplot2)
library(dplyr)
library(raster)
library(patchwork)
library(here)
setwd(here())

## Some settings.
proj_ETRSTM35 <- "+proj=utm +zone=35 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
proj_WGS84 <- "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
misval <- 0.0

## Theme.
theme_set(theme_bw())
theme_area_plot <- theme(axis.title = element_blank(),
                         axis.text = element_text(size = 5),
                         plot.margin = unit(rep(0, 4), "pt"))
LINESIZE <- 0.25
POINTSIZE <- 0.5
NORTH_ARROW_SIZE <- 0.4
NORTH_ARROW_PAD_X <- 0.08
NORTH_ARROW_PAD_Y <- 0.25

## Get polygon of study area.
study_poly <- readRDS(file.path(here(), "data", "rds", "study-area-polygon-wgs84.rds"))

## Get map of Finland.
fin <- ne_countries(scale = "large", country = "Finland", returnclass = "sf")

### NOTE: Commented out since cannot be run due to data not public. ###
## Get data.frame of datapoints.
# file <- "filter-tassu-npar16384-m3-locprior-prob0.95-diam65.48545-bd0.0015-clut0.475-plot-data.rds"
# datafolder <- file.path("data")
# inp <- readRDS(file.path(datafolder, file))
# datadf <- data.frame(x = inp[["data-x"]], 
#                      y = inp[["data-y"]])
# 
# # Get points in form needed for package sf. Need st_sfc to add CRS.
# ps <- st_sfc(st_multipoint(as.matrix(datadf)), crs = proj_ETRSTM35)
# ps_proj <- st_transform(ps, crs = proj_WGS84)

## Make plot of study area.
# 
# # Plot.
# plt_study_area <- ggplot(fin) + 
#   geom_sf(size = LINESIZE) +
#   geom_sf(data = study_poly, alpha = 0.4, fill = "blue", size = LINESIZE) +
#   geom_sf(data = ps_proj, alpha = 0.2, size = POINTSIZE) + 
#   coord_sf(ylim = c(60, 65.25))  +
#   scale_x_continuous(expand = rep(0.01, 2)) + 
#   scale_y_continuous(expand = rep(0.01, 2)) + 
#   annotation_scale(location = "tl", width_hint = 0.15, text_cex = 0.6) +
#   annotation_north_arrow(location = "tl", which_north = "true",
#                          height = unit(NORTH_ARROW_SIZE, "in"), width = unit(NORTH_ARROW_SIZE, "in"),
#                          pad_x = unit(NORTH_ARROW_PAD_X, "in"), pad_y = unit(NORTH_ARROW_PAD_Y, "in"),
#                          style = north_arrow_nautical) +
#   theme_area_plot +
#   theme(axis.text.x = element_blank(), 
#         axis.ticks.x = element_blank())

## Make plot of known territories in spring 2019.
# Get data about territories.
territory_data <- readRDS(file.path(here(), "data/rds/pack-locations-geometry-early-2019.rds"))
cntr <- st_transform(st_centroid(territory_data$geometry), proj_WGS84)
geom <- st_transform(territory_data$geometry, proj_WGS84)

# Plot.
plt_territories <- ggplot(fin) +
  geom_sf(size = LINESIZE) +
  geom_sf(data = geom, fill = "transparent") +
  geom_sf(data = cntr, size = LINESIZE) +
  coord_sf(ylim = c(60, 65.25)) +
  scale_x_continuous(expand = rep(0.01, 2)) + 
  scale_y_continuous(expand = rep(0.01, 2)) + 
  annotation_scale(location = "tl", width_hint = 0.15, text_cex = 0.6) +
  annotation_north_arrow(location = "tl", which_north = "true", 
                         height = unit(NORTH_ARROW_SIZE, "in"), width = unit(NORTH_ARROW_SIZE, "in"),
                         pad_x = unit(NORTH_ARROW_PAD_X, "in"), pad_y = unit(NORTH_ARROW_PAD_Y, "in"),
                         style = north_arrow_nautical) +
  theme_area_plot
ggsave("output/plots/territories.pdf", plt_territories, width = 6, height = 6)

## Make continental map plot.
# Get map of world.
world <- ne_countries(scale = "large", returnclass = "sf")

# Plot.
plt_continental <- ggplot(world) + 
  geom_sf(size = LINESIZE) +
  geom_sf(data = study_poly, alpha = 0.4, fill = "blue", size = LINESIZE) +
  coord_sf(ylim = c(37, 72), xlim = c(-10, 35)) +
  scale_x_continuous(expand = rep(0.01, 2)) + 
  scale_y_continuous(expand = rep(0.01, 2)) + 
  annotation_scale(location = "tl", width_hint = 0.15, text_cex = 0.6) +
  annotation_north_arrow(location = "tl", which_north = "true", 
                         height = unit(NORTH_ARROW_SIZE, "in"), width = unit(NORTH_ARROW_SIZE, "in"),
                         pad_x = unit(NORTH_ARROW_PAD_X, "in"), pad_y = unit(NORTH_ARROW_PAD_Y, "in"),
                         style = north_arrow_nautical) +
  theme_area_plot + 
  theme(plot.margin = margin(0, 0, 0, 3))
ggsave("output/plots/continental.pdf", plt_continental, width = 6, height = 6)