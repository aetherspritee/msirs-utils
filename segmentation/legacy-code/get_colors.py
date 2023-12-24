#!/usr/bin/env python3
import glob, re, skimage.io
import numpy as np
from matplotlib import pyplot as plt

USER = "dusc"
# database_directory = f"/home/{USER}/segmentation/segmented"
database_directory = f"/Users/{USER}/segmentation/"
file_names = glob.glob(database_directory + "/*.png", recursive=False)

file_names = [i for i in file_names if re.findall("map",i)]
colors = []
for file in file_names:
    img = skimage.io.imread(file)
    plt.imshow(img)
    plt.title(file)
    plt.show()

#     c = np.unique(img)
#     colors.append([i for i in c if i not in colors])


# print(colors)
data = {
    "aec": (31,119,180),
    "ael": (174,199,232),
    "cli": (255,127,14),
    "rid": (197,176,213),
    "fsf": (152,223,138), #DONE
    "sfe": (196,156,148),
    "fsg": (214,39,40),
    "fse": (44,160,44),
    "fss": (255,152,150),
    "cra": (255,187,120),
    "sfx": (227,119,194), #DONE
    "mix": (227,119,194),
    "rou": (140,86,74), #DONE
    "smo": (247,182,210),
    "tex": (127,127,127) #DONE
}
