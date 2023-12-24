#!/usr/bin/env python3

from skimage import exposure, io
from matplotlib import pyplot as plt


if __name__ == "__main__":
    img_path = "/Users/dusc/segmentation/contrast_test4.png"
    img = io.imread(img_path)
    fig = plt.figure()
    plt.imshow(img)
    img_adapteq = exposure.equalize_adapthist(img, clip_limit=0.03)
    plt.show()
    plt.imshow(img_adapteq)
    plt.show()
