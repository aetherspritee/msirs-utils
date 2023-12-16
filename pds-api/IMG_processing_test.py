#!/usr/bin/env python3
import skimage, json
from matplotlib import pyplot as plt
import numpy as np
from numba import jit


# @jit()
def process_IMG(file_path: str, img_name=None):
    with open(file_path, "rb") as f:
        text = f.readlines()

    print(len(text))
    test_bed = text[-1]
    i0 = 0
    for i in range(len(test_bed)):
        if test_bed[i] != 0:
            i0 = i
            break
    test_bed = test_bed[i0:]
    img = []
    for i in range(len(test_bed)):
        b = test_bed[i : i + 1]
        b = int.from_bytes(b)
        img.append(b)

    img = np.array(img)
    if img_name == None:
        img_name = "image"
    reformatting_img(img, img_name)
    return img


def reformatting_img(img: np.ndarray, name: str) -> None:
    print(img.shape)
    y_size = 5056
    padding = y_size - np.mod(img.shape[0], y_size)
    print(padding)
    x_size = int((img.shape[0] + padding) / y_size)
    pad = [0] * padding
    img = np.hstack((img, pad))
    print(img.shape)
    img = np.reshape(img, (x_size, y_size))
    img = img / np.max(img)
    img = skimage.color.gray2rgb(img)
    plt.imsave(f"{name}.jpg", img)


def extract_meta_data(file_path: str):
    meta_dict = {}
    with open(file_path, "r") as f:
        text = f.readlines()

    print(len(text))
    i0 = 0
    for i in range(len(text)):
        print(text[i])
        if text[i] == "END\n":
            i0 = i
            break
    metadata = text[0 : i0 - 1]
    for entry in metadata:
        key = entry.split("=")[0][:-1]
        val = entry.split("=")[1][:-1]
        if val[0] == " ":
            val = val[1:]
        try:
            val = json.loads(val)
        except:
            pass
        meta_dict[key] = val

    with open(file_path.split(".")[0] + ".json", "a+") as f:
        json.dump(meta_dict, f)
    return meta_dict


if __name__ == "__main__":
    # file_path = "/Users/dusc/msirs-utils/pds-api/P13_006213_2187_XN_38N340W.IMG"
    # file_path = "/Users/dusc/msirs-utils/pds-api/P13_006213_2644_XN_84N009W.IMG"
    # file_path = "/Users/dusc/msirs-utils/pds-api/P13_006213_1561_XN_23S332W.IMG"
    # file_path = "/Users/dusc/msirs-utils/pds-api/P13_006210_1434_XN_36S248W.IMG"
    # file_path = "/Users/dusc/msirs-utils/pds-api/P13_006211_2038_XN_23N283W.IMG"
    file_path = "/Users/dusc/msirs-utils/pds-api/P13_006213_1561_XN_23S332W.IMG"
    # stored_file_path = "/Users/dusc/msirs-utils/pds-api/test.npy"
    # img = process_IMG(file_path=file_path, img_name="image1")

    meta = extract_meta_data(file_path=file_path)
    print(meta)
