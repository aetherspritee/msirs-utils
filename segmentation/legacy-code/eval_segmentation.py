#!/usr/bin/env python3
import glob, re, os
from pathlib import Path
from matplotlib import pyplot as plt
import skimage.io

# find database folder
# get all real images
# get all mrf images
# display them side by side

home = str(Path.home())
database_directory = f"{home}/segmentation/segmented/"
print(f"{database_directory}")
# file_names = glob.glob(database_directory + "/*.jpg", recursive=True)
file_names = os.listdir(database_directory)
print(file_names)
# mrf_imgs = [i for i in file_names if re.findall("mrf", i)]
mrf_imgs = []
real_imgs = [i for i in file_names if re.findall("og_img", i)]
for og_img in real_imgs:
    check = [
        i for i in file_names if re.findall(og_img[:-10], i) and re.findall("mrf", i)
    ]
    if len(check) > 0:
        mrf_imgs.append(check[0])

print(len(real_imgs))
print(len(mrf_imgs))
print(mrf_imgs[0], real_imgs[0])

for i in range(len(mrf_imgs)):
    fig = plt.figure(figsize=(10, 7))
    print(mrf_imgs[i])
    # setting values to rows and column variables
    rows = 2
    columns = 2
    fig.add_subplot(rows, columns, 1)

    # showing image
    plt.imshow(skimage.io.imread(database_directory + mrf_imgs[i]))
    plt.axis("off")
    plt.title("Segmented")

    # Adds a subplot at the 2nd position
    fig.add_subplot(rows, columns, 2)

    # showing image
    plt.imshow(skimage.io.imread(database_directory + real_imgs[i]))
    plt.axis("off")
    plt.title("Real Image")
    plt.show()
