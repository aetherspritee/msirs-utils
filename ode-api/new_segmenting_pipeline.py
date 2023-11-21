#!/usr/bin/env python3

from main import only_download_images
from segment_images import rectify_image
from chunkify_img import cutout_image, chunkification
import numpy as np
import os, re
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = 78256587200
ImageFile.LOAD_TRUNCATED_IMAGES = True

num = 10000
west_lon = np.linspace(20.00, 340.00, num)
east_lon = [i + 10 for i in west_lon]

min_lat = np.linspace(-82.00, 80.00, num)
max_lat = [i + 2 for i in min_lat]
# comb = zip(west_lon, east_lon, min_lat, max_lat)

lons = zip(west_lon, east_lon)
lats = zip(min_lat, max_lat)

# choose desired chunk sizes
chunk_x = 1000
chunk_y = 1000

dir = "/Users/dusc/domars_benchmark/"

image_names = os.listdir(dir)
image_names = [dir + i for i in image_names if re.findall("tiff", i)]
image_names = [image_names[0]]
rect_files = []
for file in image_names:
    ret = rectify_image(file)
    if isinstance(ret, str):
        rect_files.append(ret)


# TODO: this should be done in a single loop for efficiency but for now id rather have a safe checkpoint
# in case the program crashes
# for file in rect_files:
#     # chunkification
#     _, cutout_name = cutout_image(file)
#     chunkification(cutout_name, chunk_x, chunk_y)
