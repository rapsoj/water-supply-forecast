#!/usr/bin/env python3

import os
import requests
import warnings

from tqdm import tqdm

warnings.filterwarnings("ignore")

START_YEAR = 1981
END_YEAR = 2023
YEARS = [y for y in range(START_YEAR, END_YEAR)]

ROOT = 'https://climate.arizona.edu/data/UA_SWE/'
LOCATION = 'data/swann'

def main():

    os.makedirs(LOCATION, exist_ok=True)

    for yr in tqdm(YEARS):
        fn = f'UA_SWE_Depth_WY{yr}.nc'
        url = ROOT + '/' + fn
        target = os.path.join(LOCATION, fn)
        with open(target, 'wb') as f:
            response = requests.get(url, verify=False)
            f.write(response.content)


if __name__ == '__main__':
    main()
