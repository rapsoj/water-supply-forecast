#!/usr/bin/env python3

import os
import warnings

import rioxarray
import numpy as np
import pandas as pd
import geopandas as gpd

from tqdm import tqdm

warnings.filterwarnings("ignore")

START_YEAR = 1981
END_YEAR = 2023
YEARS = [y for y in range(START_YEAR, END_YEAR)]
LOCATION = 'data/swann'

poly = gpd.read_file("data/geospatial.gpkg")
poly = poly.to_crs(6933)


def main():

    swe_volumes = []
    for yr in tqdm(YEARS):

        fn = os.path.join('data/swann', f'UA_SWE_Depth_WY{yr}.nc')
        try:
            ds = rioxarray.open_rasterio(fn, variable='SWE', mask_and_scale=True)
        except IOError:
            continue


        # Perform an initial clip to reduce the size of the array
        ds = ds.rio.write_crs(4326)
        ds = ds.rio.reproject("EPSG:6933")

        # # Calculate pixel area
        # x_pixel_size = ds.rio.transform()[0]
        # y_pixel_size = abs(ds.rio.transform()[4])
        # pixel_area = x_pixel_size * y_pixel_size # m2
        ds['SWE'] /= 1000           # mm -> m
        # ds['SWE'] *= pixel_area     # Convert to volume (m3)

        ds = ds.fillna(0)
        ds = ds.rio.write_crs(6933)
        ds_clipped_rgn = ds.rio.clip(poly.geometry.values)

        for i in range(0, len(poly)):

            # Clip to current polygon
            ds_clipped_poly = ds_clipped_rgn.rio.clip([poly.iloc[i].geometry])

            # Set -999 to nan then convert to dataframe
            # ds_clipped_poly = ds_clipped_poly.where(ds_clipped_poly['SWE'] != -999)
            df = ds_clipped_poly.to_dataframe()
            df = df.dropna()
            df = df.reset_index()

            # Group by time and average over grid cells
            # swe_vol = df.groupby(['time'])['SWE'].sum()
            swe_vol = df.groupby(['time'])['SWE'].mean()

            # Format dataframe
            swe_vol = pd.DataFrame(swe_vol).reset_index()
            swe_vol['site_id'] = poly.iloc[i].site_id
            swe_vol = swe_vol.set_index(['site_id', 'time'])
            swe_vol = swe_vol.rename({'SWE': 'SWE_depth_m'}, axis=1)
            swe_volumes.append(swe_vol)

    # Concatenate dataframes and write output
    swe_volumes = pd.concat(swe_volumes, axis=0)
    swe_volumes.to_csv("data/swann/swann_swe.csv")


if __name__ == '__main__':
    main()

# # PROCESSING:
# ds = xarray.open_dataset(fn, decode_coords='all')
# da = ds['SWE']
# weightmap = xagg.pixel_overlaps(da, poly, subset_bbox=False)
# aggregated = xagg.aggregate(da, weightmap)
# df = aggregated.to_dataframe()
#!/usr/bin/env python3
