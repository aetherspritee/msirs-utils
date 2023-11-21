#!/usr/bin/env python3
import skimage.io
import numpy as np

img = skimage.io.imread("ESP_048212_0990_COLOR_2415_11496_3415_12496.png")

cutout = img[
    # int(np.shape(img)[0] / 2 - 25) : int(np.shape(img)[0] / 2 + 25),
    24,
    int(np.shape(img)[1] / 2 - 25) : int(np.shape(img)[1] / 2 + 25),
]
print(cutout)

# あああああああああああああああああああああああ
