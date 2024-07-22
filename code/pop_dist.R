# This file computes distribution of population density by commune
# Last updated: 4 July 2024

library(sp)
library(rworldmap)
library(ggplot2)
library(raster)
library(dplyr)
library(devtools)
library(proj4)
library(measurements)
library(pryr)


# load raster
#dbfolder <- "/media/hd1/Dropbox/"
args = commandArgs(trailingOnly=TRUE)
orangefolder <- args[1] # location passed as argument
dbfolder <- paste(orangefolder, "Databases/", sep="")
spatialfolder <- paste(dbfolder, "spatial/", sep="")
spatialinfrfolder <- paste(spatialfolder, "infrastructure/spatial/", sep="")
processed_data_path <- paste(orangefolder, "data/", sep="")
popfile <- paste(spatialinfrfolder,"population_density2015/gpw_v4_population_density_adjusted_to_2015_unwpp_country_totals_rev10_2015_30_sec.tif", sep="")
countries_file <- paste(spatialinfrfolder,"countries/ne_10m_admin_0_countries.shp", sep="")
muni_file <-paste(spatialfolder,"communes/n_com_fla_000.shp", sep="")
muni_list_file <-paste(processed_data_path,"market_ROF.csv", sep="")


# load files
pop_dens <- raster(popfile)
countries <-shapefile(countries_file)
munis <-shapefile(muni_file)
muni_list <- read.csv(muni_list_file, header = TRUE)
fr_shape <- countries[countries$ADMIN=='France',]

#plot(munis)

# Only get info for communes in our sample (if do all, creates a problem for communes w/o shape data)
muni_list <- muni_list[muni_list$market != 0,]


# get polygons for sample counties
getmuni <- function(code){
  muni_poly <- munis[munis$insee_comm == code,]
  return(muni_poly)
}

sample_munis <- sapply(muni_list$codeinsee, getmuni)
sample_munisF <- SpatialPolygons(lapply(sample_munis, function(x){x@polygons[[1]]}))


# take a look at sample counties
#plot(sample_munisF)
#plot(fr_shape, xlim=c(-5,10),ylim=c(41,52), add = TRUE)



# select raster values for each sample commune
getPopDensDist <- function(poly){
  temp <- extract(pop_dens,poly)
  return(temp)
}

pDensDist <- sapply(sample_munis, getPopDensDist)



# take contraharmonic mean by commune
chmean <- function(list){
  sqmean <- mean(list*list)
  lmean <- mean(list)
  out <- sqmean/lmean
  return(out)
}
normalmean <- function(list){
  lmean <- mean(list)
  return(lmean)
}

pDensCHMeans <- sapply(pDensDist, chmean)
pDensMeans <- sapply(pDensDist, normalmean)


# output csv file
muni_list$chmean <- pDensCHMeans
muni_list$normalmean <- pDensMeans

write.csv(muni_list, file = paste(processed_data_path,"PopDensMeans.csv",sep=""))
