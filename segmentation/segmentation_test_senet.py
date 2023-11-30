#!/usr/bin/env python3

from senet_model import SENet
import skimage


if __name__ == "__main__":
    model = SENet("fullAdaptedSENetNetmodel.keras")

    img_path = (
        "/Users/dusc/codebase-v1/data/data/test/ael/B08_012727_1742_XN_05S348W_CX1593_CY12594.jpg",
    )

    img = skimage.io.imread(
        "/Users/dusc/codebase-v1/data/data/test/ael/B08_012727_1742_XN_05S348W_CX1593_CY12594.jpg"
    )
    print(img.shape)
    img = skimage.transform.resize(
        img,
        (
            img.shape[0] + 1,
            img.shape[1] + 1,
        ),
        anti_aliasing=True,
    )
    model.segment_image(img)
