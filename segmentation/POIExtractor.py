import skimage
from sklearn.cluster import DBSCAN
import re, time, copy
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from senet_model import SENet
import seaborn as sns
from PIL import Image, ImageFile

# this is needed to load larger images
Image.MAX_IMAGE_PIXELS = 78256587200
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMAGE_THRESHOLD_CERT = 11
ROI_THRESHOLD_CERT_LOW = 2
ROI_THRESHOLD_CERT_HIGH = 10
HOME = str(Path.home())
classes = {
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

color_info = {
    "aec": (31, 119, 180),
    "ael": (174, 199, 232),
    "cli": (255, 127, 14),
    "rid": (197, 176, 213),
    "fsf": (152, 223, 138),
    "sfe": (196, 156, 148),
    "fsg": (214, 39, 40),
    "fse": (44, 160, 44),
    "fss": (255, 152, 150),
    "cra": (255, 187, 120),
    "sfx": (227, 119, 194),
    "mix": (148, 103, 189),
    "rou": (140, 86, 74),
    "smo": (247, 182, 210),
    "tex": (127, 127, 127),
}


interesting_classes = [
    color_info["aec"],
    color_info["ael"],
    color_info["cli"],
    color_info["rid"],
    color_info["fsf"],
    color_info["sfe"],
    color_info["fsg"],
    color_info["fse"],
    color_info["cra"],
    color_info["sfx"],
]


class POI_Extractor:
    def __init__(self, model: SENet) -> None:
        self.model = model

    def find_POIs(self, segmented_image_path: str):
        # TODO: well this thing is kinda important
        # how the fuck did i almost forget that i need this lmfao
        segmented_image = skimage.io.imread(segmented_image_path)

    def find_batch_POIS(self, segmented_images_path: list[str]):
        # TODO: use parallelization/batching here to make it more efficient
        pass

    def process_image(self, file: str, model_stuff):
        model = model_stuff[0]
        data_transform = model_stuff[1]
        device = model_stuff[2]

        rois = []
        descriptors = []

        og_file = re.sub("mrf", "img", file)
        og_img = skimage.io.imread(og_file)
        og_img = skimage.color.rgb2gray(og_img[:, :, :3])
        # img_cert = check_img(og_file, model, data_transform, device)
        # og_file_name = og_file.split("/")[-1]
        # TODO: kinda useless for large images, delete this
        img_cert = 0
        if img_cert > IMAGE_THRESHOLD_CERT:
            # set higher requirements for ROI cert
            ROI_THRESHOLD_CERT = ROI_THRESHOLD_CERT_HIGH
        else:
            ROI_THRESHOLD_CERT = ROI_THRESHOLD_CERT_LOW

        start = time.monotonic()
        boxes = self.detect_regions(file)
        print(f"Created {len(boxes)} boxes in {time.monotonic()-start}s")
        for c in boxes:
            cutout = og_img[int(c[0]) : int(c[1]), int(c[2]) : int(c[3])]
            # print("Done with cutout")
            # TODO: implement size filter and size adjustment in the bounding box creation
            if np.shape(cutout)[0] >= 50 and np.shape(cutout)[1] >= 50:
                # run through network
                cert, pred = self.check_cutout(og_file, c)
                print("Cert: ", cert)
                # plt.imshow(og_img[int(c[0]) : int(c[1]), int(c[2]) : int(c[3])])
                # plt.title(f"{ cert }")
                # plt.show()

                if cert > ROI_THRESHOLD_CERT:
                    # save ROI
                    rois.append(c)
                    # FIXME: think about how this can work elegantly
                    desc = self.model.get_descriptor(
                        og_img[int(c[0]) : int(c[1]), int(c[2]) : int(c[3])]
                    )
                    desc = desc
                    descriptors.append(desc)

                    plt.imsave(
                        f"extracted/{classes[pred[0]]}_{cert}.png",
                        skimage.color.gray2rgb(
                            og_img[int(c[0]) : int(c[1]), int(c[2]) : int(c[3])]
                        ),
                    )
                # print("Done with chunk.")
        return rois, descriptors

    def detect_regions(self, img_path: str):
        print("Entering detect_region method...")
        img = skimage.io.imread(img_path)
        print("Loaded image")
        # fig = plt.figure()
        # plt.imshow(img)
        # plt.title("og")
        # plt.show()
        start = time.monotonic()
        # unique_vals = np.unique(img.reshape(-1, img.shape[2]), axis=0)
        int_vals = interesting_classes
        boxes = []
        for val in int_vals:
            if val[:3] in img[:, :, :3]:
                try:
                    start = start_col = time.monotonic()
                    c_img = copy.deepcopy(img)[:, :, :3]
                    print(f"Created copy of img in {time.monotonic()-start}s")
                    start = time.monotonic()
                    # c_img = np.where(
                    #     c_img != np.array(list(val)), np.array(list(val)), np.array([0, 0, 0])
                    # )
                    c_img = np.all(c_img == val, axis=-1)
                    print(f"Initial recoloring took {time.monotonic()-start}s")
                    plt.imsave("recoloring_test_1.png", c_img.astype("uint8"))
                    pix_to_cluster = []
                    start = time.monotonic()
                    pix_to_cluster = np.where(c_img == True)
                    pix_to_cluster_arr = copy.deepcopy(pix_to_cluster)
                    pix_to_cluster = [
                        [pix_to_cluster[0][i], pix_to_cluster[1][i]]
                        for i in range(len(pix_to_cluster[0]))
                    ]
                    print(f"Extracted pixels to cluster in {time.monotonic()-start}s")
                    start = time.monotonic()
                    optics = DBSCAN(metric="cityblock", eps=1, min_samples=5)
                    print(f"Done with clustering in {time.monotonic()-start}")
                    res = optics.fit(np.array(pix_to_cluster)).labels_
                    res_labels = np.unique(res)
                    print(f"LABELS: { res_labels }")

                    print(f"Finished to bounding box in {time.monotonic()-start_col}s!")
                    if len(res_labels) > 1:
                        l_img = np.zeros(np.shape(img))[:, :, :3]
                        palette = sns.color_palette(None, len(res_labels))
                        seg_colors = []
                        start = time.monotonic()
                        # TODO: this doesnt work yet
                        # colors = [[res[i]] * 3 for i in range(len(pix_to_cluster))]
                        # l_img[pix_to_cluster_arr] = colors
                        # plt.imsave("wow_much_optimization.png", l_img.astype("uint8"))
                        for idx in range(len(pix_to_cluster)):
                            if res[idx] >= 0:
                                color = [res[idx] + 1] * 3
                                if color not in seg_colors:
                                    seg_colors.append(color)
                                l_img[
                                    pix_to_cluster[idx][0], pix_to_cluster[idx][1], :
                                ] = color
                        print(f"Final recoloring took {time.monotonic()-start}s")

                        print(len(seg_colors))
                        for sc in seg_colors:
                            start = time.monotonic()
                            box = self.find_bounding_box(l_img, sc)
                            print(f"Found bounding box in {time.monotonic()-start}s")
                            if box[1] - box[0] > 0 and box[3] - box[2] > 0:
                                boxes.append(box)

                    else:
                        # only one segment, create bounding box
                        start = time.monotonic()
                        c_img2 = np.zeros(np.shape(img))
                        c_img2[:, :, 0] = c_img * val[0]
                        c_img2[:, :, 1] = c_img * val[1]
                        c_img2[:, :, 2] = c_img * val[2]
                        print(c_img2[0:4, 0:4, :])
                        box = self.find_bounding_box(c_img2, val)
                        print(f"Found bounding box in {time.monotonic()-start}s")
                        if box[1] - box[0] > 0 and box[3] - box[2] > 0:
                            boxes.append(box)
                except Exception:
                    pass
        return boxes

    def extract_regions(self, img_path, boxes, interesting_classes, color_info):
        img_path_og = re.sub("mrf", "og_img", img_path)
        img = skimage.io.imread(img_path_og)
        mrf = skimage.io.imread(img_path)
        cutouts = []
        region_info = []
        for box in boxes:
            patch = mrf[int(box[0]) : int(box[1]), int(box[2]) : int(box[3])]

            if tuple(patch[0, 0, :][:-1]) in interesting_classes:
                cutouts.append(
                    img[int(box[0]) : int(box[1]), int(box[2]) : int(box[3]), :]
                )
                region_info.append(
                    list(color_info.keys())[
                        list(color_info.values()).index((tuple(patch[0, 0, :][:-1])))
                    ]
                )
        return cutouts, region_info

    def check_cutout(self, file: str, box):
        # FIXME: replace with new model class here
        cert = 0.0
        ctx_image = Image.open(file).convert("RGB")
        ctx_image = np.array(ctx_image)[
            int(box[0]) : int(box[1]), int(box[2]) : int(box[3])
        ]
        ctx_image = data_transform(Image.fromarray(ctx_image))
        test_img1 = ctx_image.unsqueeze(0)
        vec_rep = model(test_img1.to(device))
        pred = torch.argmax(vec_rep, dim=1).cpu().detach().numpy()
        cert = torch.max(vec_rep).cpu().detach().numpy()
        # cert = float(vec_rep.cpu()[pred])
        return cert, pred

    def find_bounding_box(self, img, val):
        # TODO: pass all vals and loop over them in here for better performance
        # builds a box around an area of a certain color (denoted by val)
        img_x = np.shape(img)[1]
        img_y = np.shape(img)[0]

        idx = np.where(
            (img[:, :, 0] == val[0])
            & (img[:, :, 1] == val[1])
            & (img[:, :, 2] == val[2])
        )
        min_x = np.min(idx[0])
        max_x = np.max(idx[0])
        min_y = np.min(idx[1])
        max_y = np.max(idx[1])
        # print(f"Found bounding box: {min_x},{max_x},{min_y},{max_y}")
        box_size_x = max_x - min_x
        box_size_y = max_y - min_y
        border_x = 0
        border_y = 0
        if box_size_x < 50 or box_size_y < 50:
            # too small dont use
            # gets filtered out in detect_region method
            return [1, 0, 1, 0]
        # elif box_size_x < 100 or box_size_y < 100:
        #     border_x = 200 - box_size_x
        #     border_y = 200 - box_size_y
        elif box_size_x < 200 or box_size_y < 200:
            border_x = 200 - box_size_x
            border_y = 200 - box_size_y

        if min_y >= border_y:
            min_y = min_y - border_y
        else:
            min_y = 0
        if min_x >= border_x:
            min_x = min_x - border_x
        else:
            min_x = 0
        if max_x < img_x - border_x:
            max_x = max_x + border_x
        else:
            max_x = img_x
        if max_y < img_y - border_y:
            max_y = max_y + border_y
        else:
            max_y = img_y

        return [min_x, max_x, min_y, max_y]
