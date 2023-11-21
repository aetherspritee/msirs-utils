#!/usr/bin/env python3
from pathlib import Path
import os, re
import numpy as np
import skimage.io
from skimage.feature import hog
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

if __name__ == "__main__":
    features = []
    feature_dict = {}
    home = str(Path.home())
    base_path = home+"/Dropbox/Stuff/Code/LSIR/data/data/test/"
    categories = ["aec",   "ael",  "cli" , "cra",  "fse",  "fsf",  "fsg",  "fss" , "mix",  "rid"  ,"rou"  ,"sfe"  ,"sfx"  ,"smo" ,"tex"]
    imp_cat = ["aec",   "ael",  "cli" , "cra",  "fse",  "fsf",  "fsg",  "fss" , "sfe"  ,"sfx" , "rid"]
    meh_cat = ["rou"  ,"smo" ,"tex", "mix"]

    for category in categories:
        path = base_path+category
        # print(path)
        files = os.listdir(path)
        # print(path)
        files = [i for i in files if re.findall("jpg", i) or re.findall("png", i)]

        for file in files:
            img = skimage.io.imread(path+"/"+file)
            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

            # img_name = "test_strip_full.png"

            fd = hog(img, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1))
            features.append(fd)
            if category not in feature_dict:
                feature_dict[category] = []
            feature_dict[category].append(fd)


    pca = PCA(n_components=3)
    pca.fit(np.array(features))
    pca_feature_dict = {}
    for key in feature_dict.keys():
        pca_feature_dict[key] = pca.transform(feature_dict[key])


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    for feat in pca_feature_dict.keys():
        ax.scatter(pca_feature_dict[feat][:,0], pca_feature_dict[feat][:,1], pca_feature_dict[feat][:,2])

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
