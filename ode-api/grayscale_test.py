#!/usr/bin/env python3
import skimage.io
from skimage.color import rgb2gray
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

# img_path = "ESP_048212_0990_COLOR_2415_11496_3415_12496.png"
img_path = "ESP_048212_0990_COLOR_2415_11496_3415_12496.png"
img = skimage.io.imread(img_path)
# img[:,:,0] = (0.5*img[:,:,1]+ 0.5*img[:,:,2])
img[:,:,0] = np.zeros(np.shape(img[:,:,0]))
img = rgb2gray(img)
fig = plt.figure()
plt.imshow(img)
plt.show()
