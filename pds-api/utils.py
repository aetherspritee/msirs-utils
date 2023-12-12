#!/usr/bin/env python3
import skimage
from matplotlib import pyplot as plt
import numpy as np


def process_IMG(file_path: str):
    A = np.fromfile(file_path, dtype="int16", sep="")

    with open(file_path, "r") as f:
        text = f.readlines()

    print(len(text))
    print(text[0:-2])
    test_bed = text[-1][20000:20020]
    print(test_bed.encode("utf-8"))

    # with open("test.jpg", "ab") as f:
    #     f.write(text[-1])
    # return img


if __name__ == "__main__":
    file_path = "/Users/dusc/msirs-utils/pds-api/CRU_000001_9999_XN_99N999W.IMG"

    img = process_IMG(file_path=file_path)
    # plt.imshow(img)
    # plt.show()
