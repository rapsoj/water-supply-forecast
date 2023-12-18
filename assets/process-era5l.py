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
MONTHS = [m for m in range(1, 13)]
LOCATION = 'data/ERA5L'

poly = gpd.read_file("data/geospatial.gpkg")
poly = poly.to_crs(6933)


def main():

    df_list = []
    for yr in tqdm(YEARS):
        for month in MONTHS:
            year_str = str(yr).zfill(4)
            month_str = str(month).zfill(2)
            fn = os.path.join(LOCATION, 'download_' + year_str + '_' + month_str, 'data.nc')

            try:
                ds = rioxarray.open_rasterio(fn, mask_and_scale=True)
            except IOError:
                continue

            ds = ds.rio.write_crs(4326)
            ds = ds.rio.reproject("EPSG:6933")

            # Calculate pixel area
            # x_pixel_size = ds.rio.transform()[0]
            # y_pixel_size = abs(ds.rio.transform()[4])
            # pixel_area = x_pixel_size * y_pixel_size # m2
            # ds['SWE'] /= 1000           # mm -> m
            # ds['SWE'] *= pixel_area     # Convert to volume (m3)

            # Perform an initial clip to reduce the size of the array
            ds_clipped_rgn = ds.rio.clip(poly.geometry.values)

            for i in range(0, len(poly)):

                # Clip to current polygon
                ds_clipped_poly = ds_clipped_rgn.rio.clip([poly.iloc[i].geometry])

                # Convert to dataframe
                df = ds_clipped_poly.to_dataframe()
                df = df.dropna()
                df = df.reset_index()
                df = df.drop(['x', 'y', 'spatial_ref'], axis=1)

                # Compute windspeed from u and v components
                df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2)
                df = df.drop(['u10', 'v10'], axis=1)

                # Group by time and average
                df_mean = df.groupby(['time']).mean()

                # Give variables meaningful names
                df_mean = df_mean.rename({
                    'd2m': '2m_dewpoint_temperature_K',
                    'sp': 'surface_air_pressure_Pa',
                    'tp': 'total_precipitation_m',
                    'e': 'evaporation_m',
                    'sf': 'snowfall_mwe',
                    'smlt': 'snowmelt_mwe',
                    'ro': 'runoff_m',
                    'sd': 'snow_depth_mwe',
                    'pev': 'potential_evaporation_m',
                    't2m': '2m_temperature_K',
                    'swvl1': 'volumetric_soil_water_layer1_m3_per_m3',
                    'swvl2': 'volumetric_soil_water_layer2_m3_per_m3',
                    'swvl3': 'volumetric_soil_water_layer3_m3_per_m3',
                    'swvl4': 'volumetric_soil_water_layer4_m3_per_m4',
                    'ssr': 'surface_net_solar_radiation_J_per_m2',
                    'str': 'surface_net_thermal_radiation_J_per_m2'
                }, axis=1)

                tm = df_mean.index[0]
                variables = list(df_mean.columns)
                period_start = pd.to_datetime(tm.strftime("%Y-%m-%d"))
                period_end = period_start + pd.offsets.MonthEnd(0)
                df_mean['period_start'] = period_start
                df_mean['period_end'] = period_end
                df_mean = df_mean.reset_index() #drop=True)
                df_mean = df_mean[['time', 'period_start', 'period_end'] + variables]

                df_mean['site_id'] = poly.iloc[i].site_id
                df_mean = df_mean.set_index(['site_id', 'time'])
                df_list.append(df_mean)


    # Concatenate dataframes and write output
    df = pd.concat(df_list, axis=0)
    df.to_csv("data/ERA5L/era5l.csv")


if __name__ == '__main__':
    main()
