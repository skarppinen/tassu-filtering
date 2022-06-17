library(RColorBrewer)
library(ggplot2)

# Defaults for various sizes in plots.
SIZE_DEFAULTS <- list(
  axis_text_size = 5,
  axis_title_size = 5,
  strip_text_size = 5,
  legend_text_size = 5,
  legend_title_size = 5
)

# Default theme for faceted plot.
THEME_FACETED <- with(SIZE_DEFAULTS, {
  theme_bw() +
    theme(panel.spacing = unit(1, "pt"),
          plot.margin = unit(c(0, 2, 0, 2), "pt"),
          axis.text = element_text(size = axis_text_size),
          axis.title = element_text(size = axis_title_size),
          strip.text = element_text(size = strip_text_size),
          legend.title = element_text(size = legend_title_size),
          legend.text = element_text(size = legend_text_size))
})

# Coloring function from gray to black. 
GRAY_TO_BLACK_COLOR_FUN <- colorRampPalette(c("#C6C6C6", "black"))

# Take a vector of particle counts and return a factor which
# prepends "N = " before the number. (used for labels)
make_particle_factor_name <- function(npar, textprefix = "M = ") {
  factor(paste0(textprefix, npar),
         levels = paste0(textprefix, sort(unique(npar))))
}
