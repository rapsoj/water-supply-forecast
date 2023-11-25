#!/usr/bin/env python3

import os
import pandas as pd
import re
import click

from tqdm import tqdm

# See this paper for a statistical model that predicts AMJJ volume using SWE
# https://doi.org/10.1175/JHM-D-21-0229.1

# These could also be args
MIN_RECORD_LENGTH = 40
MIN_STATIONS_PER_SITE = 1
TEST_PERIOD = [y for y in range(2005, 2024)]
START_YEAR = TEST_PERIOD[-1] - MIN_RECORD_LENGTH + 1

@click.command()
@click.option('--input-directory', '-d', default='../assets/data', type=click.Path(exists=True), help='Path to the assets data directory')
@click.option('--start', default=1984, help='Start year')
@click.option('--end', default=2023, help='End year')
@click.option('--output', '-o', default='snotel.csv', help='Output file with merged SNOTEL data')
def main(input_directory, start, end, output):
    snotel_directory = os.path.abspath(os.path.join(input_directory, "snotel"))
    if not os.path.exists(snotel_directory):
        raise ValueError(f'Directory {input_directory} does not contain SNOTEL data')

    # Get list of years for which data is available
    available_years = [int(re.sub('^FY', '', f)) for f in os.listdir(snotel_directory) if re.match('^FY[0-9]{4}$', f)]
    available_years.sort()
    years = [yr for yr in range(int(start), int(end) + 1)]
    if not all([yr in available_years for yr in years]):
        raise ValueError(f'Insufficient data: expecting data for years {start}-{end}')

    # Loop through each year and retrieve the available station files
    data_record = []
    for yr in years:
        location = os.path.join(snotel_directory, "FY" + str(yr))
        if not os.path.exists(location):
            raise ValueError(f'Location {location} does not exist!')
        fs = os.listdir(location)   # TODO filter with regex
        ids = [f.split('_')[0] for f in fs]
        data_record.append(pd.DataFrame({'YEAR': yr, 'FILES': fs, 'ID': ids}))

    # Identify locations with at least 30 years of data from the
    # end of the hindcast period
    data_record = pd.concat(data_record)

    # Check that each site has only one file associated with it
    n_files_per_year = data_record.groupby(['YEAR', 'ID'])['FILES'].count().rename('N_FILES').reset_index()
    if any(n_files_per_year['N_FILES']) > 1:
        raise ValueError

    # Filter record length
    stations = data_record.loc[(data_record['YEAR'] >= START_YEAR)]
    stations = stations.groupby('ID')['YEAR'].size().rename('N_YEAR').reset_index()
    stations = stations.loc[(stations['N_YEAR'] >= MIN_RECORD_LENGTH)]

    # Link to sites to snotel stations metadata
    sites_to_snotel_stations = pd.read_csv(os.path.join(snotel_directory, "sites_to_snotel_stations.csv"))
    site_ids = [i.split(':')[0] for i in sites_to_snotel_stations['stationTriplet']]
    sites_to_snotel_stations['ID'] = site_ids
    stations = stations.merge(sites_to_snotel_stations, on = 'ID')

    # Limit by number of SNOTEL stations per site
    n_stations_per_site = stations.groupby(['site_id'])['ID'].count().rename('N_STATIONS').reset_index()
    n_stations_per_site = n_stations_per_site.loc[n_stations_per_site['N_STATIONS'] > MIN_STATIONS_PER_SITE]
    stations = stations.loc[stations['site_id'].isin(n_stations_per_site['site_id'])]

    # Get all sites
    site_ids = stations['site_id'].unique()
    data_list = []
    for site_id in tqdm(site_ids):
        station_ids = stations.loc[stations['site_id'] == site_id]['ID']
        for station_id in station_ids:
            for yr in years:
                data_record_id = data_record.loc[(data_record['ID'] == station_id) & (data_record['YEAR'] == yr)]
                if data_record_id.shape[0] == 0:
                    next
                fn = data_record_id['FILES'].iloc[0]
                location = os.path.join(snotel_directory, "FY" + str(yr))
                x = pd.read_csv(os.path.join(location, fn), parse_dates = [0])
                x = x.set_index('date')
                # Make sure we have complete timeseries
                x_reindexed = x.reindex(
                    pd.date_range(
                        start=x.index.min(),
                        end=x.index.max(),
                        freq='1D'
                    )
                )
                # The only situation in which this would be needed is if data is missing for the 1 April of every year
                x_reindexed = x_reindexed.interpolate(method='linear')
                x_reindexed.index.name = 'date'
                x_reindexed = x_reindexed.loc[(x_reindexed.index.month == 4) & (x_reindexed.index.day == 1)]
                x_reindexed = x_reindexed.reset_index()
                x_reindexed['site'] = site_id
                x_reindexed['station'] = station_id
                index_varnames = ['date', 'site', 'station']
                varnames = [c for c in x if re.match('.*_DAILY$', c)]
                x_reindexed = x_reindexed[index_varnames + varnames]
                data_list.append(x_reindexed)

    data = pd.concat(data_list)
    data.to_csv(output, index=False)


if __name__ == "__main__":
    main()
