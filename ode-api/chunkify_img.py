#!/usr/bin/env python3

import numpy as np
import skimage.io, os
from skimage.color import rgb2gray
from PIL import Image, ImageFile
from matplotlib import pyplot as plt

Image.MAX_IMAGE_PIXELS = 78256587200
ImageFile.LOAD_TRUNCATED_IMAGES = True
# fig = plt.figure()
# plt.imshow(img)
# plt.show()
def cutout_image(img_path: str,ups=50):
    img = skimage.io.imread(img_path)
    img = rgb2gray(img)
    cutoff = int(0.15*np.shape(img)[0])
    max_x0 = -1
    min_x0 = 10e10
    max_x1 = -1
    min_x1 = 10e10
    for i in range(np.shape(img)[0]-cutoff):
        l1 = img[i,:]
        try:
            x = [i for i in range(len(l1)) if int(10*l1[i]) != 0]
            x0 = x[0]
            x1 = x[-1]
            print(f"{x0} :: {x1}")
            print(f"== {x1-x0}")
            if x0 > max_x0:
                max_x0 = x0
            if x1 < min_x1:
                min_x1 = x1
        except Exception:
            print("uf")

    print(min_x0, max_x0)
    eps = max_x0 - min_x0 + ups
    print(min_x1, max_x1)

    # calc diff of shift, use as border.
    cropped_img = img[0:-cutoff,max_x0+eps:max_x1-eps]
    fig = plt.figure()
    plt.imshow(cropped_img)
    plt.title(np.shape(cropped_img))
    plt.show()
    # TODO: im not sure whether thisll work, check this thoroughly later
    img_name = img_path.split(".")[0]
    img_name = img_name[:-4]
    print(img_name+"cutout.png")
    new_p = Image.open(img_path)
    new_p = new_p.crop((max_x0+eps,0, max_x1-eps, np.shape(img)[0]-cutoff))
    new_p.save(img_name+"cutout.png")
    return cropped_img, img_name+"cutout.png"

def chunkification(img_path, chunk_size_x, chunk_size_y):
    # make sure the image name ends in cutout or it will be messed up lmao
    cropped_img = skimage.io.imread(img_path)
    # croppedImg = Image.open(img_path)
    # grayscale conversion
    cropped_img = grayscale_conversion(cropped_img)
    croppedImg = Image.fromarray(cropped_img)
    print(np.shape(cropped_img))
    if chunk_size_x > np.shape(cropped_img)[0]:
        print(f"Desired chunksize x is too large for image. {chunk_size_x =}, {np.shape(cropped_img)[0] =}")
        raise Exception
    if chunk_size_y > np.shape(cropped_img)[1]:
        print(f"Desired chunksize y is too large for image. {chunk_size_y =}, {np.shape(cropped_img)[1] =}")
        raise Exception
    chunks_per_row = int(np.shape(cropped_img)[1]/chunk_size_y)
    print(chunks_per_row)
    chunks_per_col = int(np.shape(cropped_img)[0]/chunk_size_x)
    print(chunks_per_col)
    # calculate border and place chunks in centre to avoid edge cases
    border_y = np.shape(cropped_img)[1]-chunk_size_y*chunks_per_row
    border_y = int(border_y/2)
    print(border_y)
    border_x = np.shape(cropped_img)[0]-chunk_size_x*chunks_per_col
    border_x = int(border_x/2)
    print(border_x)

    # begin with chunkification
    for x_num in range(1,chunks_per_col+1):
        for y_num in range(1,chunks_per_row+1):
            x0 = int((x_num-1)*chunk_size_x+border_x)
            x1 = int(x_num*chunk_size_x+border_x)
            y0 = int((y_num-1)*chunk_size_y+border_y)
            y1 = int(y_num*chunk_size_y+border_y)
            print((y0,x0,y1,x1))

            # TODO: im not sure whether thisll work, check this thoroughly later
            path = img_name.split(".")[0]
            path = path[:-7]

            isExist = os.path.exists(path)
            isExist = os.path.exists(path)
            if not isExist:
                os.makedirs(path)
            if not isExist:
                os.makedirs(path)

            cutout = croppedImg.crop((y0,x0,y1,x1))
            # cutout = grayscale_conversion(cutout)
            cutout = Image.fromarray((255*np.array(cutout)).astype(np.uint8))
            cutout.save(path+f"/{path}_{y0}_{x0}_{y1}_{x1}.png")

def grayscale_conversion(img):
    if isinstance(img, str):
        img = skimage.io.imread(img)
    else:
        if not isinstance(img, np.ndarray):
            img = np.array(img)

    for i in range(3):
        if not 0 in img[:,:,i]:
            # channel is uncorrupted, goooood
            pass
        else:
            # no bueno, delete channel
            img[:,:,i] = np.zeros(np.shape(img[:,:,i]))

    # grayscale image
    img = rgb2gray(img)
    return img

if __name__ == "__main__":
    # img_name = "ESP_048212_0990_COLOR_rect.png"
    # cutout = cutout_image(img_name)
    img_name = "ESP_048212_0990_COLOR_cutout.png"
    chunkification(img_name, chunk_size_x=1000, chunk_size_y=1000)
