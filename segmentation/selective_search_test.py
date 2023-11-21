#!/usr/bin/env python3

from selective_search import selective_search, box_filter
from skimage.io import imread
import skimage.data
from matplotlib import pyplot as plt
from matplotlib import patches as mpatches
import numpy as np

img1 = imread("test_strip_full.png")
img = np.zeros((np.shape(img1)[0], np.shape(img1)[1], 3))
# img[:, :, 0] = img1
# img[:, :, 1] = img1
# img[:, :, 2] = img1
img = img1[:,:,:3]
# img = imread("/Users/agony/codebase-v1/CTX_stripe_densenet1611_img.png")[:, :, :3]
# img = skimage.data.astronaut()
boxes = selective_search(img, mode="quality", random_sort=False)
boxes_filter = box_filter(boxes, min_size=20, topN=80)

fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(img)
for x1, y1, x2, y2 in boxes_filter:
    bbox = mpatches.Rectangle(
        (x1, y1), (x2 - x1), (y2 - y1), fill=False, edgecolor="red", linewidth=1
    )
    ax.add_patch(bbox)

plt.axis("off")
plt.show()
