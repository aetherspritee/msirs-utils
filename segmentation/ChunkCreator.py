import tensorflow as tf
import numpy as np
import skimage

IMAGE_SIZE = 224


class ChunkCreator(tf.keras.utils.Sequence):
    def __init__(
        self,
        img: np.ndarray,
        window_size: int = IMAGE_SIZE,
        step_size: int = 2,
        batch_size: int = 64,
    ):
        self.batch_size = batch_size
        self.image_full = img
        self.cs = 0
        self.window_size = window_size

        print(self.image_full.shape)

        # Get shapes of "new" full image
        self.image_size_full = np.shape(self.image_full)

        self.num_tiles_full = np.ceil(
            np.array(self.image_size_full) / self.window_size
        ).astype("int")
        print(f"{self.num_tiles_full = }")

        wd = self.image_size_full[0]
        hd = self.image_size_full[1]
        # create new image of desired size and color (blue) for padding
        print(window_size)
        print(self.num_tiles_full)
        ww, hh = window_size * self.num_tiles_full
        # hh = window_size * self.num_tiles_full[1]

        # compute center offset
        xx = (ww - wd) // 2
        yy = (hh - hd) // 2

        # copy img image into center of result image
        self.padded_full = np.zeros(
            tuple((self.num_tiles_full * self.window_size).astype("int")),
            dtype=np.uint8,
        )
        self.padded_full[xx : xx + wd, yy : yy + hd] = self.image_full

        # self.padded_full[:self.image_size_full[0], :self.image_size_full[1]] = self.image_full

        step_size_full = step_size
        idx_tiles_full_a = np.rint(
            np.arange(0, self.num_tiles_full[0] * self.window_size, step_size_full)
        ).astype("int")
        idx_tiles_full_b = np.rint(
            np.arange(0, self.num_tiles_full[1] * self.window_size, step_size_full)
        ).astype("int")

        self.idx_tiles_full_a = idx_tiles_full_a[
            idx_tiles_full_a + self.window_size
            < self.num_tiles_full[0] * self.window_size
        ]
        self.idx_tiles_full_b = idx_tiles_full_b[
            idx_tiles_full_b + self.window_size
            < self.num_tiles_full[1] * self.window_size
        ]

        self.num_full = np.array(
            [self.idx_tiles_full_a.__len__(), self.idx_tiles_full_b.__len__()]
        )
        self.out_shape = (
            self.idx_tiles_full_a.__len__(),
            self.idx_tiles_full_b.__len__(),
        )

    def __len__(self):
        return np.prod(self.num_full)

    def __getitem__(self, idx):
        images = []
        centers = []

        low = idx * self.batch_size
        high = min(
            low + self.batch_size,
            self.num_full[0] * self.num_full[1],
        )
        for my_idx in range(low, high):
            # print(f"{my_idx = }")
            idx_aa, idx_bb = np.unravel_index(my_idx, self.num_full)
            idx_a = self.idx_tiles_full_a[idx_aa]
            idx_b = self.idx_tiles_full_b[idx_bb]
            image = self.padded_full[
                idx_a : idx_a + self.window_size, idx_b : idx_b + self.window_size
            ]
            centers.append(image[self.window_size // 2, self.window_size // 2])
            image = np.dstack([image] * 3)
            image = skimage.transform.resize(
                image, (IMAGE_SIZE, IMAGE_SIZE, 3), anti_aliasing=True
            )
            image = np.resize(image, (1, IMAGE_SIZE, IMAGE_SIZE, 3))
            images.append(image)

        images = np.resize(np.array(images), (len(images), IMAGE_SIZE, IMAGE_SIZE, 3))
        centers = np.resize(np.array(centers), (len(centers), 1))
        return images, centers
