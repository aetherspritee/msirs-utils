import numpy as np
import skimage
import tensorflow as tf
import scipy
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

# network only accepts images of size 224x224x3
IMAGE_SIZE = 224


class SENet:
    def __init__(self, model_path: str) -> None:
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, img: np.ndarray) -> str:
        """
        Returns string of predicted class for given input image
        """
        image = self.prep_image(img)
        predictions = self.model.predict(image)
        prediction = predictions[0]
        return CATEGORIES[np.argmax(prediction)]

    def get_descriptor(self, img: np.ndarray) -> np.ndarray:
        """
        Returns 512 dimensional descriptor of image, by using output of second to last model layer
        """
        extractor = tf.keras.Model(
            inputs=self.model.inputs, outputs=self.model.layers[610].output
        )
        image = self.prep_image(img)
        feature = extractor(image)
        return feature.numpy().reshape((-1,))

    # FIXME: Test this fully
    def segment_image(
        self,
        img: np.ndarray,
        window_size: int = 200,
        step_size: int = 1,
        batch_size: int = 64,
        workers: int = 8,
    ) -> None:
        segmenter_sequence = ChunkCreator(
            img=img, window_size=window_size, step_size=step_size
        )

        print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

        self.scores = self.model.predict(
            segmenter_sequence,
            batch_size=batch_size,
            workers=workers,
            use_multiprocessing=True,
        )

        # creation of resulting segmentation maps
        img_size_x = img.shape[0]
        img_size_y = img.shape[1]

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

        n = len(CATEGORIES.keys())
        from_list = mpl.colors.LinearSegmentedColormap.from_list
        cm = from_list(None, plt.cm.tab20(range(0, n)), n)

        # TODO: check this output!!
        plt.imsave(
            "segment_test.png",
            self.predictions,
            cmap=cm,
            vmin=0,
            vmax=len(CATEGORIES.keys()),
        )

    async def vectorize(self, img: np.ndarray):
        """
        Just wraps get_descriptor() so that it can be used by weaviate for as vectorizer
        """
        return self.get_descriptor(img)

    def heatmap(self, img: np.ndarray):
        img = self.prep_image(img)
        last_layer_weights = self.model.layers[-3].get_weights()[0]

        model2 = tf.keras.models.Model(
            inputs=self.model.input,
            outputs=(self.model.layers[-8].output, self.model.layers[-3].output),
        )
        last_conv_layer_output, last_layer_output = model2.predict(img)

        last_conv_layer_output = np.squeeze(last_conv_layer_output)
        pred = np.argmax(last_layer_output)
        h = int(img.shape[1] / last_conv_layer_output.shape[1])
        # w = int(img.shape[2] / last_conv_layer_output.shape[2])
        # upsampled_heat = scipy.ndimage.zoom(
        #     last_conv_layer_output, (h, h, 1), order=1
        # )
        upsampled_heat = skimage.transform.resize(
            last_conv_layer_output,
            (
                last_conv_layer_output.shape[0] * h,
                last_conv_layer_output.shape[1] * h,
                512,
            ),
            anti_aliasing=True,
        )
        weights_for_pred = last_layer_weights[pred]
        # weights_p1 = weights_for_pred[:256]
        # print(np.shape(weights_p1))

        # TODO: upsample the last dim for conv layer to make it compatable
        # heatmap = np.dot(upsampled_heat.reshape((224 * 224, 256)), weights_p1).reshape(
        #     224, 224
        # )

        heatmap = np.dot(
            upsampled_heat.reshape((224 * 224, 512)), weights_for_pred
        ).reshape(224, 224)

        fig, ax = plt.subplots()
        ax.imshow(np.squeeze(img))
        ax.imshow(heatmap, cmap="jet", alpha=0.5)
        plt.show()

    @staticmethod
    def prep_image(img: np.ndarray) -> np.ndarray:
        """
        Prepares a greyscale image (1 channel) for use with the network
        """
        image = skimage.color.gray2rgb(img)
        image = skimage.transform.resize(
            image, (IMAGE_SIZE, IMAGE_SIZE), anti_aliasing=True
        )
        image = np.resize(image, (1, 224, 224, 3))
        return image


if __name__ == "__main__":
    # small example showcasing class

    model = SENet("fullAdaptedSENetNetmodel.keras")

    img_paths = [
        "/Users/dusc/codebase-v1/data/data/test/ael/B08_012727_1742_XN_05S348W_CX1593_CY12594.jpg",
        "/Users/dusc/codebase-v1/data/data/test/cra/B07_012260_1447_XI_35S194W_CX4750_CY4036.jpg",
        "/Users/dusc/codebase-v1/data/data/test/ael/P06_003352_1763_XN_03S345W_CX440_CY3513.jpg",
        "/Users/dusc/codebase-v1/data/data/test/cra/K01_053719_1938_XI_13N232W_CX1714_CY6640.jpg",
        "/Users/dusc/codebase-v1/data/data/test/cli/F08_038957_1517_XN_28S040W_CX727_CY4635.jpg",
        "/Users/dusc/codebase-v1/data/data/test/fss/B07_012259_1421_XI_37S167W_CX4172_CY5294.jpg",
    ]

    features = []

    for i in range(len(img_paths)):
        # features.append(model.get_descriptor(skimage.io.imread(img_paths[i])))
        print(model.predict(skimage.io.imread(img_paths[i])))
        model.heatmap(img=skimage.io.imread(img_paths[i]))

    # dis01 = scipy.spatial.distance.cosine(features[0], features[1])
    # dis02 = scipy.spatial.distance.cosine(features[0], features[2])
    # dis03 = scipy.spatial.distance.cosine(features[0], features[3])
    # dis13 = scipy.spatial.distance.cosine(features[1], features[3])
    # dis12 = scipy.spatial.distance.cosine(features[1], features[2])
    # dis10 = scipy.spatial.distance.cosine(features[1], features[0])

    # print(f"{dis01 = }")
    # print(f"{dis02 = }")
    # print(f"{dis03 = }")
    # print(f"{dis10 = }")
    # print(f"{dis12 = }")
    # print(f"{dis13 = }")
