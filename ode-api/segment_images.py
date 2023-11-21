import numpy as np
import os, re, time, vg
from PIL import Image, ImageFile
import skimage.io
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
from scipy.ndimage import affine_transform


# TODO: Test if larger eps improves results
def find_angle(img: np.ndarray, eps=30):
    # the idea is to find the outer most pixel that is not black
    # (on both sides of the strip), and then calculate the angle
    # by describing the edges of the image as a linear function
    test_lines = [100, 200, 300]
    base_vec = np.array([1, 0])
    base_vec = base_vec / np.linalg.norm(base_vec)
    angles = []
    for ln in test_lines:
        l1 = img[ln, :]
        l2 = img[ln + eps, :]
        # find the first non black pixel in there
        x0 = [i for i in range(len(l1)) if l1[i] != 0][0]
        print(f"{x0 = }")
        x1 = [i for i in range(len(l2)) if l2[i] != 0][0]
        print(f"{x1 = }")
        # build vector
        edge_vec = np.array([eps, x1 - x0])
        edge_vec = edge_vec / np.linalg.norm(edge_vec)
        print(f"{edge_vec = }")
        print(f"{base_vec = }")
        dot_product = np.dot(edge_vec, base_vec)
        print(f"{dot_product = }")
        if edge_vec[0] > 0 and edge_vec[1] > 0:
            angles.append(-np.arccos(dot_product))
        else:
            angles.append(np.arccos(dot_product))
        # find angle
    # TODO: majority decision to make sure we find the correct vector
    print(angles)
    angle = max(set(angles), key=angles.count)
    return angle


def rectify_image(file: str, eps=30):
    # do affine transformation on images strip
    print(file)
    img = Image.open(file)
    img2 = np.array(img)
    # img2 = rgb2gray(img2)
    print("loaded image lol")
    fig = plt.figure()
    plt.imshow(img)
    plt.show()

    ang = find_angle(img2, eps=eps)
    ang = np.rad2deg(ang)
    ang = -(90 - ang)
    print(ang)
    rot_img = img.rotate(ang)
    fig = plt.figure()
    plt.imshow(rot_img)
    plt.show()
    rect_file_name = file.split(".")[0] + "_rect.png"
    skimage.io.imsave(rect_file_name, np.array(rot_img))
    return rect_file_name


def find_angle2(img: np.ndarray, eps=30):
    test_lines = [100, 200, 300]
    base_vec = np.array([0, 1])
    base_vec = base_vec / np.linalg.norm(base_vec)
    angles = []
    for ln in range(1, np.shape(img)[1] + 1):
        l1 = img[:, -ln]
        x0 = [i for i in range(len(l1)) if l1[i] != 0]
        print(x0)
        if len(x0) > 0:
            b = x0[-1]
            l = np.shape(img)[1] - ln
            break
    edge_vec = np.array([b, l])
    edge_vec = edge_vec / np.linalg.norm(edge_vec)
    print(f"{edge_vec = }")
    print(f"{base_vec = }")
    dot_product = np.dot(edge_vec, base_vec)
    print(f"{dot_product = }")
    if edge_vec[0] > 0 and edge_vec[1] > 0:
        angle = -np.arccos(dot_product)
    else:
        angle = np.arccos(dot_product)
    return angle


def rectify_image2(file: str):
    try:
        # do affine transformation on images strip
        print(file)
        img = Image.open(file)
        img2 = np.array(img)
        img2 = rgb2gray(img2)
        print("loaded image lol")
        fig = plt.figure()
        plt.imshow(img)
        plt.show()

        ang = find_angle2(img2)
        ang = np.rad2deg(ang)
        ang = -ang
        print(ang)
        rot_img = img.rotate(ang)
        fig = plt.figure()
        plt.imshow(rot_img)
        plt.show()
        rect_file_name = file.split(".")[0] + "_rect.png"
        skimage.io.imsave(rect_file_name, np.array(rot_img))
        return rect_file_name

    except Exception:
        print(
            "Probably (!) all black pixels, happens sometimes no worries. But also: maybe some other error i cba to fix rn"
        )
        return -1


if __name__ == "__main__":
    all_imgs = os.listdir()
    all_imgs = [i for i in all_imgs if re.findall("JP2", i)]
    all_imgs = ["test_strip.png"]
    # ESP_030529_0990_COLOR
    # ESP_037992_1000_COLOR.JP2
    # all_imgs = [all_imgs[3]]
    Image.MAX_IMAGE_PIXELS = 78256587200
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # TODO: build function around this
    for file in all_imgs:
        rectify_image(file, eps=100)
