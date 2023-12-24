#!/usr/bin/env python3

import os, re, random
import numpy as np
import skimage.io
from process_image import post_process
from matplotlib import pyplot as plt

USER = "dusc"
segmented_dir = f"/Users/{USER}/segmentation/segmented"

files = os.listdir(segmented_dir)
# only use postprocessed images
files = [i for i in files if re.findall("mrf", i)]
w = [1]
while 1 in w:
    q_img = random.choice(files)
    q_og_img = re.sub("mrf", "og_img", q_img)
    q_pp = post_process(segmented_dir+"/"+q_img)
    w = [q_pp[i] for i in q_pp.keys()]

print("Found good query image. Beginning search..")
vals = []
img_res = []
for file in files:
    img = skimage.io.imread(segmented_dir+"/"+file)
    og_img_name = re.sub("mrf", "og_img", file)
    og_img = skimage.io.imread(segmented_dir+"/"+og_img_name)
    pp_info = post_process(segmented_dir+"/"+file)
    # print(f"{pp_info = }")

    comp = [abs(q_pp[i]-pp_info[i]) for i in q_pp.keys()]
    vals.append(np.sum(comp))
    img_res.append(og_img_name)
    # print(f"{sum(comp) = }")


ordered_imgs = [i for _, i in sorted(zip(vals, img_res))]
ordered_vals = sorted(vals)

fig = plt.figure()
rows = 2
columns = 2
fig.add_subplot(rows, columns, 1)

# showing image
plt.imshow(skimage.io.imread(segmented_dir+"/"+q_og_img))
plt.axis('off')
plt.title("Query")

# Adds a subplot at the 2nd position
fig.add_subplot(rows, columns, 2)

# showing image
plt.imshow(skimage.io.imread(segmented_dir+"/"+ordered_imgs[0]))
plt.axis('off')
plt.title(f"Best result: {np.round(ordered_vals[0])}")

fig.add_subplot(rows, columns, 3)

# showing image
plt.imshow(skimage.io.imread(segmented_dir+"/"+ordered_imgs[1]))
plt.axis('off')
plt.title(f"Second best result: {np.round(ordered_vals[1], 3)}")

fig.add_subplot(rows, columns, 4)

# showing image
plt.imshow(skimage.io.imread(segmented_dir+"/"+ordered_imgs[2]))
plt.axis('off')
plt.title(f"Third best result: {np.round(ordered_vals[2],3)}")
plt.show()
