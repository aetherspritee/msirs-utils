#!/usr/bin/env python3
from main import download_images
import numpy as np
import time

num = 10000
west_lon = np.linspace(20.00, 340.00, num)
east_lon = [i+10 for i in west_lon]

min_lat = np.linspace(-82.00, 80.00, num)
max_lat = [i+2 for i in min_lat]
#comb = zip(west_lon, east_lon, min_lat, max_lat)

lons = zip(west_lon, east_lon)
lats = zip(min_lat, max_lat)
# FIXME: This crashes due to some error in opencv; no clue why:
# error: (-215:Assertion failed) !ssize.empty() in function 'resize'
counter = 0
for wl, el  in lons:
    for min_lat,max_lat in lats:
        print(wl, el, min_lat, max_lat)
        # only use *RED* as product id, its the only one i found that yields good images
        download_images('HIRISE',product_type="RDRV11", product_id="",file_name="",western_lon=wl,eastern_lon=el, min_lat=min_lat, max_lat=max_lat, number_product_limit=100)
        counter += 1
        if counter == 100:
            time.sleep(300)
            counter = 0

