# Import packages
import sys
import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from shapely.geometry import Point

# Import French communes shapefile
loc = sys.argv[1]
gdf_communes = gpd.read_file(loc + "Databases/spatial/communes_shp/communes-20160119-shp/communes-20160119.shp")

# Import Ookla phone records
df = pd.read_stata(loc + "data/phone_records.dta")

# Add geometry data and convert to geopandas
geom = df.apply(lambda x: Point([x['client_longitude'], x['client_latitude']]), axis=1)
df = gpd.GeoDataFrame(df, geometry=geom) # geom is a Series
df.crs = {'init': 'epsg:4326'}

# Merge with French communes to indentify commune location lies within
gdf = gpd.sjoin(df, gdf_communes, how="left", op='within')

# Export to Stata dataset
df_export = pd.DataFrame(gdf.drop(columns=gdf_communes.columns[1:]))
df_export.to_stata(loc + "data/phone_records_insee.dta", write_index=False)
