#!/usr/bin/env python3
import skimage
from matplotlib import pyplot as plt
import numpy as np
from numba import jit


# @jit()
def process_IMG(file_path: str):
    A = np.fromfile(file_path, dtype="int16", sep="")

    with open(file_path, "rb") as f:
        text = f.readlines()

    print(len(text))
    print(text[0:-2])
    test_bed = text[-1]  # [20000:20020]
    print(test_bed[0])
    i0 = 0
    for i in range(len(test_bed)):
        if test_bed[i] != 0:
            i0 = i
            break
    print(test_bed[5000:5020])
    print(test_bed[5000:5001])
    testlol = test_bed[5002:5003]
    test_bed = test_bed[i0:-1]
    x_size = 5056
    y_size = int(np.floor(len(test_bed) / x_size))
    test_bed = test_bed[0 : x_size * y_size]
    print(len(test_bed) / x_size)
    print("wow", test_bed[:-20:-1])
    img = []
    for i in range(len(test_bed)):
        b = test_bed[i : i + 1]
        b = int.from_bytes(b)
        img.append(b)

    img = np.reshape(np.array(img), (x_size, y_size))
    plt.imsave("test2.jpg", img)
    # with open("test.jpg", "ab") as f:
    #     f.write(text[-1])
    # return img


if __name__ == "__main__":
    file_path = "/Users/dusc/msirs-utils/pds-api/P13_006213_2187_XN_38N340W.IMG"

    img = process_IMG(file_path=file_path)
    # plt.imshow(img)
    # plt.show()
