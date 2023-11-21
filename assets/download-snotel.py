#!/usr/bin/env python3

import os
import pandas as pd
import click

from wsfr_download import snotel

# See this paper for a statistical model that predicts AMJJ volume using SWE
# https://doi.org/10.1175/JHM-D-21-0229.1

@click.command()
@click.option('--output-directory', '-o', default='.', help='Path to download SNOTEL data')
@click.option('--start', default=1984, help='Start year')
@click.option('--end', default=2023, help='End year')
def main(output_directory, start, end):
    os.environ['WSFR_DATA_ROOT'] = output_directory
    years = [yr for yr in range(int(start), int(end) + 1)]
    data_record = []
    for yr in years:
        snotel.download_snotel([yr])


if __name__ == "__main__":
    main()
