#!/usr/bin/env python3
import shutil
import numpy as np
import weaviate, re, os, random
import skimage.io
from matplotlib import pyplot as plt
from process_image import detect_regions, extract_regions, post_process, find_bounding_box
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import glob
from domars_map import MarsModel, HIRISE_Image, segment_image
from PIL import Image
from scipy import spatial
from pathlib import Path

home_dir = str(Path.home())

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


color_info = {
    "aec": (31,119,180),
    "ael": (174,199,232),
    "cli": (255,127,14),
    "rid": (197,176,213),
    "fsf": (152,223,138), #DONE
    "sfe": (196,156,148),
    "fsg": (214,39,40),
    "fse": (44,160,44),
    "fss": (255,152,150),
    "cra": (255,187,120),
    "sfx": (227,119,194), #DONE
    "mix": (148,103,189),
    "rou": (140,86,74), #DONE
    "smo": (247,182,210),
    "tex": (127,127,127) #DONE
}



interesting_classes = [
    color_info["cra"],
    color_info["aec"],
    color_info["ael"],
    color_info["cli"],
    color_info["rid"],
    color_info["fsf"],
]


data_transform = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

hyper_params = {
    "batch_size": 64,
    "num_epochs": 15,
    "learning_rate": 1e-2,
    "optimizer": "sgd",
    "momentum": 0.9,
    "model": "densenet161",
    "num_classes": 15,
    "pretrained": False,
    "transfer_learning": False,
}


class Pipeline():
    def __init__(self, db_adr: str):
        self.client = weaviate.Client(db_adr)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        model = MarsModel(hyper_params)
        print(home_dir+"/models")
        checkpoint = torch.load(
            home_dir+"/models/" + hyper_params["model"] + ".pth", map_location=torch.device("cpu")
        )
        model.load_state_dict(checkpoint)
        model = model.to(self.device)
        self.model = model
        self.model.eval()

    def add_to_db(self, data_object: dict, vec: list):
        self.client.data_object.create(data_object, "SegmentedImg", vector=vec)

    def preliminary_search(self,vec: list,limit=10, with_distance=False):
        if not with_distance:
            vec_dict = {"vector": vec}
            result = (
                self.client.query
                .get("SegmentedImg", ["name", "region_descriptors"])
                .with_near_vector(vec_dict)
                .with_limit(limit)
                .do()
            )
        else:
            vec_dict = {"vector": vec}
            result = (
                self.client.query
                .get("SegmentedImg", ["name", "region_descriptors"])
                .with_near_vector(vec_dict)
                .with_additional(["distance"])
                .with_limit(limit)
                .do()
            )

        # res = [i["name"] for i in result["data"]["Get"][]]
        return result

    def query_image(self, file: str):
        print(f"{file = }")
        # preprocess image
        if re.findall("og_img", file):
            self.preprocess_image(file)
            print("Created segmented image")
        # process image
        file = re.sub("og_img","mrf",file)
        print(f"{file =}")
        info_dict, region_info, q_cutouts = self.process_image(file)
        info_l = [info_dict[i] for i in info_dict.keys()]

        if len(q_cutouts) > 1:
            q_descriptors = self.region_cnn_eval(q_cutouts)

            #  to data base
            # data_object = {"name": file, "regions": region_info, "cutouts": q_cutouts, "region_descriptors": q_descriptors}
            print("Processed image. Starting search...")
            # preliminary search
            results = self.preliminary_search(info_l)
            # preliminary search v2: check occurrence of interesting categories
            names = [i["name"] for i in results["data"]["Get"]["SegmentedImg"]]
            descriptors = [eval(i["region_descriptors"]) for i in results["data"]["Get"]["SegmentedImg"]]

            mean_results, single_results = self.region_comparison(q_descriptors, names, descriptors)
            vec = {"vector": info_l}
            db_check_results = self.client.query.get("SegmentedImg", ["region_descriptors"]).with_near_vector(vec).with_additional(["distance"]).with_limit(1).do()
            res = [i["_additional"]["distance"] for i in db_check_results["data"]["Get"]["SegmentedImg"]]
            if res[0] != 0:
                #  to data base
                print("Adding to database...")
                data_object = {"name": file, "region_descriptors": str(q_descriptors)}
                self.add_to_db(data_object, vec=info_l)
            else:
                print("Entry exists already, skipping...")

            return mean_results, single_results
        else:
            # no regions of interest. only do preliminary search.
            print("No interesting regions found, only doing preliminary search.")
            results = self.preliminary_search(info_l, with_distance=True)
            files = [i["name"] for i in results["data"]["Get"]["SegmentedImg"]]
            files1 = files[:int(np.round(len(files)/2))]
            files2 = files[int(np.round(len(files)/2)):]
            distances = [i["_additional"]["distance"] for i in results["data"]["Get"]["SegmentedImg"]]
            distances = [[i]*len(distances) for i in distances]
            distances1 = distances[:int(np.round(len(distances)/2))]
            distances2 = distances[int(np.round(len(distances)/2)):]
            # build fitting return values
            # mean_results = [[files1[i], distances1[i], []] for i in range(len(distances1))]
            # single_results = [[files2[i], distances2[i], []] for i in range(len(distances2))]
            mean_results = [files1, distances1, []]
            single_results = [files2, distances2, []]

            return mean_results, single_results


    def build_db(self, files: list):
        # FIXME: zero cutouts arent yet working!
        for file in files:
            info_dict, region_info, q_cutouts = self.process_image(file)
            if len(q_cutouts) > 1:
                info_l = [info_dict[i] for i in info_dict.keys()]
                q_descriptors = self.region_cnn_eval(q_cutouts)
                vec = {"vector": info_l}
                results = self.client.query.get("SegmentedImg", ["region_descriptors"]).with_near_vector(vec).with_additional(["distance"]).with_limit(1).do()
                res = [i["_additional"]["distance"] for i in results["data"]["Get"]["SegmentedImg"]]
                if res[0] != 0:
                    #  to data base
                    print("Adding to database...")
                    data_object = {"name": file, "region_descriptors": str(q_descriptors)}
                    self.add_to_db(data_object, vec=info_l)
                else:
                    print("Entry exists already, skipping...")

    def check_db(self):
        # where_filter = {
        #     "path": ["name"],
        #     "operator": "Equal",
        #     "valueText": img,
        # }

        result = (
            self.client.query
            .aggregate("SegmentedImg")
            .with_fields("meta {count}")
            # .with_where(where_filter)
            .do()
        )
        print(result)

    def region_cnn_eval(self, cutouts):
        # FIXME: Handle no cutouts case!!
        # prepare cutout:
        descriptors = []
        for cutout in cutouts:
            ctx_image = data_transform(Image.fromarray(cutout).convert('RGB'))
            # FIXME: i need to make sure the image preprocessing works correctly
            test_img1 = ctx_image.unsqueeze(0)
            vec_rep = self.model(test_img1.to(self.device))
            descriptor = self.model.fc_layer_output(test_img1.to(self.device))
            descriptor = descriptor.tolist()
            descriptors.append(descriptor)

        return descriptors

    @staticmethod
    def process_image(file: str) -> tuple[dict, list, list]:
        boxes = detect_regions(file)
        print("Found bounding boxes for query image")
        cutouts, region_info = extract_regions(file, boxes, interesting_classes, color_info)
        print("Extracted regions...")
        info_dict = post_process(file)
        return info_dict, region_info, cutouts

    def region_comparison(self, q_descriptors: list, db_files: list, db_descriptors: list, num_to_return=5):
        best_per_image = []
        for file in range(len(db_files)):
            best_curr_sim = []
            best_curr_desc = []

            print(len(db_descriptors))
            print(len(q_descriptors))
            for q_desc in q_descriptors:
                c_best_ssd = -10e10
                c_best_desc = []
                c_best_file = ""
                for desc in db_descriptors[file]:
                    print(len(desc))
                    ssd = np.sum((np.array(q_desc)-np.array(desc))**2)
                    if ssd > c_best_ssd:
                        c_best_ssd = ssd
                        c_best_desc = desc
                best_curr_sim.append(c_best_ssd)
                best_curr_desc.append(c_best_desc)
            # have best matches within file for every q_descriptor
            best_per_image.append([best_curr_sim, best_curr_desc, db_files[file]])

        # sort by best_curr_sim in ascending order
        mean_results = []
        best_single_results = []
        files = []
        descriptors = []
        print(best_per_image)
        for db_img in range(len(best_per_image)):
            files.append(best_per_image[db_img][2])
            descriptors.extend([x for _,x in sorted(zip(best_per_image[db_img][0], best_per_image[db_img][1]))])
            ordered_best_sim = sorted(best_per_image[db_img][0])
            print(f"{ordered_best_sim = }")

            # sort via mean
            mean_results.append(np.mean(ordered_best_sim))
            # sort via single best result
            best_single_results.append(ordered_best_sim[0])


        best_means_files = [x for _,x in sorted(zip(mean_results, files))]
        best_means_descriptors = [x for _,x in sorted(zip(mean_results, descriptors))]
        best_means = sorted(mean_results)
        best_single_files = [x for _,x in sorted(zip(best_single_results, files))]
        best_single_descriptors = [x for _,x in sorted(zip(best_single_results, descriptors))]
        best_single = sorted(best_single_results)

        mean_return = [best_means_files[:num_to_return], best_means[:num_to_return], best_means_descriptors[:num_to_return]]
        single_return = [best_single_files[:num_to_return], best_single[:num_to_return], best_single_descriptors[:num_to_return]]
        return mean_return, single_return

    def visualize_results(self,query,best_mean, best_single):
        query = self.get_img_from_mrf(query)
        q_img = skimage.io.imread(query)
        num_to_show = len(best_mean)
        plt.figure()
        plt.subplot(1,2*num_to_show+1,1)
        title = query.split("/")[-1][:14]
        plt.title(f"Query: {title}")
        plt.imshow(q_img)

        for i in range(num_to_show):
            plt.subplot(1,2*num_to_show+1,i+2)
            r_img = skimage.io.imread(self.get_img_from_mrf(best_mean[0][i]))
            plt.imshow(r_img)
            title = best_mean[0][i].split("/")[-1][0:14]
            plt.title(f"{title}")

        for i in range(num_to_show):
            plt.subplot(1,2*num_to_show+1,i+2)
            r_img = skimage.io.imread(self.get_img_from_mrf(best_single[0][i]))
            plt.imshow(r_img)
            title = best_single[0][i].split("/")[-1][0:14]
            plt.title(f"{title}")
        plt.show()

    def preprocess_image(self, img: str):
        # segment given image
        file_name = img.split("/")[-1][:-24]
        step_size = 2
        plt.imsave(
            home_dir+"/segmentation/segmented/" + file_name + "_" + "densenet161" + str(step_size) + "_og_img.png",
            skimage.io.imread(img)
            )

        ctx_image = HIRISE_Image(path=img, transform=data_transform, step_size=step_size)

        test_loader = DataLoader(
            ctx_image, batch_size=64, shuffle=False, num_workers=8, pin_memory=True
        )
        print("Segmenting image for real")
        segment_image(test_loader, self.model, self.device, hyper_params, step_size=step_size, img_name=file_name)
        pass

    def store_for_ui(self, folder: str, mean_results: list, single_results: list, query: str):
        # clean up old results
        images_path = home_dir+"/segmentation/segmented/"
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")


        file = [i for i in os.listdir(images_path) if re.findall(img, images_path+i)][0]
        # FIXME: this should cause issues
        shutil.copy(images_path+file, folder+f"query{file[-4:]}")

        i1 = [i for i in mean_results[0]]
        i2 = [i for i in single_results[0]]
        results = i1 + i2
        print(results)
        # look for file in storage folder
        for result in results:
            # print("listdir",os.listdir(images_path)[0])
            # print("result",result)
            # print("combi",images_path+os.listdir(images_path)[0])
            file = [i for i in os.listdir(images_path) if re.findall(result, images_path+i)]
            print(f"{result = }")
            print(file)
            file = file[0]
            file = self.get_img_from_mrf(file)
            # FIXME: this should cause issues
            shutil.copy(images_path+file, folder+f"retrieval_{results.index(result)}{file[-4:]}")

    @staticmethod
    def clear_queue():
        folder = home_dir+"/query/"
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")

    @staticmethod
    def get_img_from_mrf(img: str):
        img = re.sub("mrf", "og_img", img)
        return img

    @staticmethod
    def get_mrf_from_img(img: str):
        img = re.sub("og_img", "mrf", img)
        return img

if __name__ == "__main__":
    pipe = Pipeline("http://localhost:8080")
    print("Instantiated pipeline")
    home_dir = str(Path.home())
    test_path = home_dir+f"/segmentation/segmented/"
    all_imgs = os.listdir(test_path)
    all_imgs = [i for i in all_imgs if re.findall("mrf", i)]
    all_imgs = [test_path+i for i in all_imgs][63:]
    # img = all_imgs
    # all_imgs = [all_imgs]
    # print(all_imgs)
    pipe.check_db()
    #pipe.build_db(all_imgs)
    #img = random.choice(all_imgs)
    #img = re.sub("mrf", "og_img", img)
    query_path = home_dir+"/query/"
    img = query_path+os.listdir(query_path)[0]
    img_name = img.split("/")[-1:][0]
    img = test_path+img_name
    print(img)

    v, f = pipe.query_image(img)
    pipe.clear_queue()
    #pipe.visualize_results(img, v, f)

    pipe.store_for_ui(home_dir+f"/server-test/",v,f, img)
