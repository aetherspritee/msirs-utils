import numpy as np
import skimage, wget, os, json, sklearn.cluster
from PIL import Image
from matplotlib import pyplot as plt
from ODE_retrieval_utils import Query

Image.MAX_IMAGE_PIXELS = 933120000


class CTX:
    def __init__(self) -> None:
        self.url = ""

    def query(self, query_dict: dict) -> list:
        Q = Query()
        results = Q.query(query=query_dict)
        return results

    @staticmethod
    def is_corrupted(img_path: str) -> bool:
        corrupted = False

        with open(img_path, "r") as f:
            text = f.readlines()

        for entry in text:
            if entry.split("=")[0][:-1] == "DATA_QUALITY_DESC":
                if entry.split("=")[1][2:-2] != "OK":
                    print(entry.split("=")[1][2:-2])
                    corrupted = True
        return corrupted

    def download(self, to_download: list, output_directory: str = os.getcwd()) -> list:
        download_paths = []
        for item in to_download:
            file = wget.download(item, out=output_directory)
            print(f"Downloading {file}")
            if self.is_corrupted(file):
                # corrupted data, delete file
                print(f"Error: Data corrupted; Deleting file {file}")
                os.remove(file)
            else:
                download_paths.append(file)

        return download_paths

    def get(self, query: dict, output_directory: str = "./"):
        results = self.query(query_dict=query)
        download_paths = self.download(results)
        for item in download_paths:
            _ = self.process_CTX_IMG(item)

    def process_CTX_IMG(self, file_path) -> np.ndarray:
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

        img_name = file_path.split(".")[-2]
        self.reformatting_img(img, img_name)
        return img

    def post_process(self, images: list[str]):
        for img_path in images:
            img = skimage.io.imread(img_path)
            img = skimage.color.rgb2gray(img)
            line1 = img[0:5, :]
            line2 = img[-1:-5, :]
            # try sobel filter first? then apply threshold? should be 2 very distinct lines use some density kernel mybe?
            print(f"{img.shape = }")
            sobel_img1 = skimage.filters.sobel_v(line1)
            # sobel_img1 = skimage.color.gray2rgb(sobel_img1)
            plt.imshow(sobel_img1)
            plt.show()
            skimage.io.imsave("line.jpg", sobel_img1)
            MS = sklearn.cluster.MeanShift()
            preds = MS.fit_predict(line1)
            print(preds)

    def post_process_proto(self, image: str):
        img = skimage.io.imread(image)
        img = skimage.color.rgba2rgb(img)
        img = skimage.color.rgb2gray(img)
        plt.imshow(img)
        plt.show()
        print(img.shape)
        MS = sklearn.cluster.MeanShift()
        preds = MS.fit_predict(img)
        print(preds)

    # TODO: do i want to add connection to other msirs tools here as well? Currently dont think so, as id rather
    # import these tools into the others and integrate them that way around

    @staticmethod
    def extract_meta_data(file_path: str, output_directory: str = ""):
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

        if output_directory == "":
            output_directory = file_path.split(".")[0] + ".json"
        else:
            if output_directory[-1] != "/":
                output_directory = output_directory + "/"
            output_directory = output_directory + file_path.split("/")[-1]
        with open(output_directory, "a+") as f:
            json.dump(meta_dict, f)
        return meta_dict

    @staticmethod
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


QUERY_DICT = {
    "target": "mars",
    "mission": "MRO",
    "instrument": "CTX",
    "product_type": "EDR",
    "western_lon": "",
    "eastern_lon": "",
    "min_lat": "",
    "max_lat": "",
    "min_ob_time": "",
    "max_ob_time": "",
    "product_id": "P13_00621*",
    "query_type": "",
    "results": "",
    "number_product_limit": "1",
    "result_offset_number": "10",
    "file_name": "*.IMG",
}


if __name__ == "__main__":
    ctx = CTX()
    # ctx.get(QUERY_DICT)
    ctx.post_process_proto("/Users/dusc/line.png")
