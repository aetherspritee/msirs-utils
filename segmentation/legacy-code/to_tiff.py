#!/usr/bin/env python3
from PIL import Image
import numpy as np

path = "ESP_046128_2465_RED_img_row_11264_col_3072_w_1024_h_1024_x_0_y_0_densenet1612_og_img.png"

im = Image.open(path)
print(np.shape(np.array(im)))
# im.save(path[:-4] + ".tiff")
