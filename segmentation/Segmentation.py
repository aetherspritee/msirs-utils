import skimage
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from ChunkCreator import ChunkCreator

CATEGORIES = {
    0: "aec",
    1: "ael",
    2: "cli",
    3: "cra",
    4: "fse",
    5: "fsf",
    6: "fsg",
    7: "fss",
    8: "mix",
    9: "rid",
    10: "rou",
    11: "sfe",
    12: "sfx",
    13: "smo",
    14: "tex",
}


class Segmentation:
    def __init__(self, model: tf.keras.Models) -> None:
        self.model = model

    # TODO: test this!!!!
    def segment_image(
        self,
        image: np.ndarray,
        window_size: int = 200,
        step_size: int = 1,
        batch_size: int = 64,
        workers: int = 8,
        post_process: bool = True,
        borders: list[int] = [100, 50, 100, 50],
        output_directory: str = "./segmented/",
        name: str = "test_lmfao.png",
    ) -> tuple:
        # does the segmentation for a single image

        if self.check_image(image):
            segmenter_sequence = ChunkCreator(
                img=image, window_size=window_size, step_size=step_size
            )

            print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

            self.scores = self.model.predict(
                segmenter_sequence,
                batch_size=batch_size,
                workers=workers,
                use_multiprocessing=True,
            )

            # creation of resulting segmentation maps
            img_size_x = image.shape[0]
            img_size_y = image.shape[1]

            self.scores = np.reshape(
                self.scores,
                (
                    int(img_size_x / step_size),
                    int(img_size_y / step_size),
                    len(CATEGORIES.keys()),
                ),
            )
            self.scores = skimage.transform.resize(
                self.scores, (img_size_x, img_size_y, len(CATEGORIES.keys()))
            )
            self.predictions = np.zeros(self.scores.shape)
            for x in range(self.scores.shape[0]):
                for y in range(self.scores[1]):
                    self.predictions[x, y] = np.argmax(self.scores[x, y])
            self.predictions = skimage.transform.resize(
                self.predictions, (img_size_x, img_size_y)
            )

            if post_process:
                self.post_process_segmentation(borders)

            n = len(CATEGORIES.keys())
            from_list = mpl.colors.LinearSegmentedColormap.from_list
            cm = from_list(None, plt.cm.tab20(range(0, n)), n)

            if output_directory[-1] != "/":
                output_directory += "/"

            name = name.split("/")[-1]

            # TODO: check this output!!
            plt.imsave(
                output_directory + name,
                self.predictions,
                cmap=cm,
                vmin=0,
                vmax=len(CATEGORIES.keys()),
            )
            # TODO: save rest of the images

            return self.scores, self.predictions
        else:
            return ()

    def find_POIs(self):
        # TODO: well this thing is kinda important
        # how the fuck did i almost forget that i need this lmfao
        pass

    def batch_predict(self) -> None:
        # TODO: i want this feature
        pass

    # TODO: i want more possiblilities here, minor variations of the segment_image method mainly im thinking

    def check_image(
        self, image: np.ndarray, window_size: int = 200, max_border: int = 100
    ) -> bool:
        # TODO: implement basic checks for images to be segmented, i.e make sure sizes are ok
        # kinda like this but spend some more time thinking bout it ya twat
        if (
            image.shape[0] > window_size + 2 * max_border
            and image.shape[1] > window_size + 2 * max_border
        ):
            return True
        else:
            return False

    def post_process_segmentation(self, borders: list[int]):
        self.predictions = self.predictions[
            borders[0] : -borders[2], borders[1] : -borders[3]
        ]
        # TODO: add all maps here
