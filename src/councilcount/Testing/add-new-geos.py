import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# This file adds congressional and NYS assembly districts to the 2020 BBL Population Estimate file.

bbl_path = "/Users/LLopez-Jensen/Documents/GitHub/councilcount-py/src/councilcount/data/bbl-population-estimates_2020.csv"
assembly_path = "/Users/LLopez-Jensen/Documents/GitHub/councilcount-py/src/councilcount/data/assembly district-nyc-wide.geojson"
congress_path = "/Users/LLopez-Jensen/Documents/GitHub/councilcount-py/src/councilcount/data/congressional district-nyc-wide.geojson"

df = pd.read_csv(bbl_path)

gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
assembly = gpd.read_file(assembly_path)
congress = gpd.read_file(congress_path)

# Ensure same CRS
assembly = assembly.to_crs(gdf.crs)
congress = congress.to_crs(gdf.crs)

gdf = gpd.sjoin(gdf, assembly[["geometry", "assembly district"]], how="left", predicate="within")
gdf = gdf.drop(columns=["index_right"], errors="ignore")
gdf = gpd.sjoin(gdf, congress[["geometry", "congressional district"]], how="left", predicate="within")
gdf = gdf.drop(columns=["index_right"], errors="ignore")

assert gdf["assembly district"].notna().all(), "Missing values found in assembly_district!"
# All good on the assembly front!

# We expect some missing congressional districts: didn't provide geometries 
# for 3 or 16 since most of these districts' populations are outside of NYC.
missing_congress = gdf[gdf["congressional district"].isna()]
missing_congress["assembly district"].value_counts()
# All of those assembly districts are in Northern Bronx or Eastern Queens and overlap with the excluded CDs. Looks good!

# Clean up district columns to match ground truth
gdf["assembly district"] = gdf["assembly district"].astype(int)
gdf["congressional district"] = (gdf["congressional district"].astype("Int64"))
# Save
gdf.drop(columns="geometry").to_csv(
    "/Users/LLopez-Jensen/Documents/GitHub/councilcount-py/src/councilcount/data/puma-bbl-population-estimates_2020.csv",
    index=False
)