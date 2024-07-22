# This file gets population densities around base stations
# Last updated: 4 July 2024

library(sp)
library(rworldmap)
library(ggplot2)
library(raster)
library(dplyr)
library(devtools)
library(proj4)
# devtools::install_github("dkahle/ggmap")
library(ggmap)
library(measurements)
library(pryr)


# http://sedac.ciesin.columbia.edu/data/set/gpw-v4-population-density-rev10
# http://thematicmapping.org/downloads/world_borders.php

#dbfolder <- "/media/hd1/Dropbox/"
args = commandArgs(trailingOnly=TRUE)
orangefolder <- args[1] # location passed as argument
dbfolder <- paste(orangefolder, "Databases/", sep="")
spatialinfrfolder <- paste(dbfolder, "spatial/infrastructure/spatial/", sep="")
processed_data_path <- paste(orangefolder, "data/", sep="")
antfile <- paste(processed_data_path, "ant_coordinates.csv", sep="")
popfile <- paste(spatialinfrfolder,"population_density2015/gpw_v4_population_density_adjusted_to_2015_unwpp_country_totals_rev10_2015_30_sec.tif", sep="")
borderfile <- paste(spatialinfrfolder,"TM_WORLD_BORDERS-0.3/TM_WORLD_BORDERS-0.3.shp", sep="")
countries_file <- paste(spatialinfrfolder,"countries/ne_10m_admin_0_countries.shp", sep="")

# load files
ant_points <- read.csv(antfile, header = TRUE)
pop_dens <- raster(popfile)
borders <- shapefile(borderfile)
countries <-shapefile(countries_file)

# get coordinates in decimal degrees
ant_points$lat <- paste(ant_points$cor_nb_dg_lat,ant_points$cor_nb_mn_lat,ant_points$cor_nb_sc_lat,sep=" ")
ant_points$lon <- paste(ant_points$cor_nb_dg_lon,ant_points$cor_nb_mn_lon,ant_points$cor_nb_sc_lon,sep=" ")

ant_points$lat = measurements::conv_unit(ant_points$lat, from = 'deg_min_sec', to = 'dec_deg')
ant_points$lon = measurements::conv_unit(ant_points$lon, from = 'deg_min_sec', to = 'dec_deg')

ant_points$lat <- as.numeric(ant_points$lat)
ant_points$lat <- ifelse(ant_points$cor_cd_ns_lat=="N",ant_points$lat,-ant_points$lat)

ant_points$lon <- as.numeric(ant_points$lon)
ant_points$lon <- ifelse(ant_points$cor_cd_ew_lon=="E",ant_points$lon,-ant_points$lon)


ant_points_sp <- ant_points
coordinates(ant_points_sp) <- ~lon+lat

# for each antenna, get population density from raster
ant_points_sp$pop_dens <- extract(pop_dens,ant_points_sp)


fr_shape <- countries[countries$ADMIN=='France',]
fr_shape <- spTransform(fr_shape, crs(pop_dens))
crs(ant_points_sp) <- crs(fr_shape)

crs(borders)
crs(pop_dens)
crs(fr_shape)
crs(ant_points_sp)

# have a look
#plot(pop_dens,xlim=c(-5,10),ylim=c(41,52))
#plot(borders[borders$NAME == 'France',], add = TRUE)
#plot(fr_shape, add = TRUE)

# focus on metro france
ant_points_cropped <- ant_points_sp[fr_shape,]

# export antenna data
write.csv(ant_points_cropped, file = paste(processed_data_path,"ant_with_pop.csv", sep=""))
