#!/usr/bin/env python3

import os
import sys
import signal
import datetime

import cdsapi
import xarray

from pathlib import Path
from zipfile import ZipFile


def get_previous_month():
    today = datetime.date.today()
    first = today.replace(day=1)
    last_month = first - datetime.timedelta(days=1)
    return last_month


START_YEAR = 1980
last_month = get_previous_month()
END_YEAR = last_month.year
END_MONTH = last_month.month
YEARS = [str(y) for y in range(START_YEAR, END_YEAR + 1)]
MONTHS = [str(m).zfill(2) for m in range(1, 13)]
VARIABLES = [
    '2m_dewpoint_temperature',
    '2m_temperature',
    'volumetric_soil_water_layer_1',
    'volumetric_soil_water_layer_2',
    'volumetric_soil_water_layer_3',
    'volumetric_soil_water_layer_4',
    'surface_net_solar_radiation',
    'surface_net_thermal_radiation',
    '10m_u_component_of_wind',
    '10m_v_component_of_wind',
    'surface_pressure',
    'total_precipitation',
    'total_evaporation',
    'snowfall',
    'snowmelt',
    'runoff',
    'snow_depth_water_equivalent',
    'potential_evaporation',
]
# VARIABLE_SHORT_NAME = {
#     '2m_dewpoint_temperature': 'd2m',
#     '2m_temperature': 't2m',
#     'volumetric_soil_water_layer_1': 'swvl1',
#     'volumetric_soil_water_layer_2': 'swvl2',
#     'volumetric_soil_water_layer_3': 'swvl3',
#     'volumetric_soil_water_layer_4': 'swvl4',
#     'surface_net_solar_radiation': 'ssr',
#     'surface_net_thermal_radiation': 'str',
#     '10m_u_component_of_wind': 'u10',
#     '10m_v_component_of_wind': 'v10',
#     'surface_pressure': 'sp',
#     'total_precipitation': 'tp',
#     'total_evaporation': 'e',
#     'snowfall',
#     'snowmelt',
#     'runoff',
#     'snow_depth_water_equivalent': 'sd',
#     'potential_evaporation': 'pev'
# }
DESTINATION = './data/ERA5L'


c = cdsapi.Client()


def main():

    os.makedirs(DESTINATION, exist_ok=True)

    break_flag = False
    for year in YEARS:
        for month in MONTHS:
            fn = os.path.join(
                DESTINATION, f'download_{year}_{month}.netcdf.zip'
            )
            if not os.path.isfile(fn):
                try:
                    c.retrieve(
                        'reanalysis-era5-land-monthly-means',
                        {
                            'product_type': 'monthly_averaged_reanalysis',
                            'variable': VARIABLES,
                            'year': year,
                            'month': month,
                            'time': '00:00',
                            'area': [
                                52, -123, 35,
                                -104
                            ],
                            'format': 'netcdf.zip',
                        },
                        fn
                    )

                except KeyboardInterrupt:
                    print("Program interrupted by user")
                    break_flag = True
                    break

                except:
                    # Otherwise
                    pass

            if os.path.isfile(fn):
                exdir = os.path.join(DESTINATION, f'download_{year}_{month}')
                with ZipFile(fn, 'r') as f:
                    f.extractall(exdir, members = ['data.nc'])

            #     ds = xarray.open_dataset(os.path.join(exdir, 'data.nc'))
            #     for var in variables:
            #         varname = VARIABLE_SHORT_NAME[var]
            #         try:
            #             da = ds[varname]
            #         except KeyError:
            #             continue

            #         year_str = str(year).zfill(4)
            #         month_str = str(month).zfill(2)
            #         fn = os.path.join(
            #             dest,
            #             f'era5land_{year_str}{month_str}_{varname}.nc'
            #         )
            #         if os.path.isfile(fn):
            #             os.remove(fn)
            #         da.to_netcdf(fn)

        if break_flag:
            break


if __name__ == '__main__':
    main()


# # EXPERIMENT
# import xagg
# import xarray

# x = xarray.open_dataset("../decadal-flood-prediction/seasonal-prediction-data-preparation/ERA5L/era5land_198001_tp.nc")
# x = x.rename_dims({'latitude': 'lat', 'longitude': 'lon'})
# x = x.rename_vars({'latitude': 'lat', 'longitude': 'lon'})
# # x = x.rename({'longitude': 'lon'})
# poly = gpd.read_file("data/geospatial.gpkg")
# weightmap = xagg.pixel_overlaps(x, poly)
