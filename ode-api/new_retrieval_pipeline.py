#!/usr/bin/env python3

from main import only_download_images
from segment_images import rectify_image
from chunkify_img import cutout_image, chunkification
import numpy as np
import time
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = 78256587200
ImageFile.LOAD_TRUNCATED_IMAGES = True

num = 10000
west_lon = np.linspace(20.00, 340.00, num)
east_lon = [i+10 for i in west_lon]

min_lat = np.linspace(-82.00, 80.00, num)
max_lat = [i+2 for i in min_lat]
#comb = zip(west_lon, east_lon, min_lat, max_lat)

lons = zip(west_lon, east_lon)
lats = zip(min_lat, max_lat)

# choose desired chunk sizes
chunk_x = 1000
chunk_y = 1000

# FIXME: This crashes due to some error in opencv; no clue why:
# error: (-215:Assertion failed) !ssize.empty() in function 'resize'
counter = 0
image_names = []
for wl, el  in lons:
    for min_lat,max_lat in lats:
        print(wl, el, min_lat, max_lat)
        # downloads images in JP2 format and returns list of file names
        # TODO: check for image on disk to avoid duplicate downloads!!
        image_names.append(only_download_images('HIRISE',product_type="RDRV11", product_id="",file_name="",western_lon=wl,eastern_lon=el, min_lat=min_lat, max_lat=max_lat, number_product_limit=100))
        counter += 1
        if counter == 100:
            time.sleep(300)
            counter = 0

# TODO: add the custom processing pipeline here
rect_files = []
for file in image_names:
    ret = rectify_image(file)
    if isinstance(ret, str):
        rect_files.append(ret)


# TODO: this should be done in a single loop for efficiency but for now id rather have a safe checkpoint
# in case the program crashes
for file in rect_files:
    # chunkification
    _, cutout_name = cutout_image(file)
    chunkification(cutout_name, chunk_x, chunk_y)
