#!/usr/bin/env python3

from pathlib import Path
import os, re, glob
import numpy as np
import skimage.io
from skimage.feature import hog
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from skimage.feature import shape_index, hessian_matrix, hessian_matrix_eigvals, hog
from skimage.filters import sobel_h, sobel_v, sobel
from sklearn import mixture

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# TODO: test binary imp/not imp distinction, somehow make edge orientation histograms viable, try both classes and binary

HOME = str(Path.home())
PATH = HOME + "/codebase-v1/data/data/test/"
VAL_PATH = HOME + "/codebase-v1/data/data/val/"


def my_shape_index(image, sigma=1):
    H = hessian_matrix(image, sigma=sigma, order="rc", use_gaussian_derivatives=True)
    l1, l2 = hessian_matrix_eigvals(H)

    # don't warn on divide by 0 as occurs in the docstring example
    with np.errstate(divide="ignore", invalid="ignore"):
        return (2.0 / np.pi) * np.arctan((l2 + l1) / (l2 - l1)), np.sqrt(
            np.divide((np.square(l1) + np.square(l2)), 2)
        )


@ignore_warnings(category=ConvergenceWarning)
def shape_feature_extraction(file, subdir=PATH, bins=9):
    # print(file)
    img_path = glob.glob(subdir + "/**/" + file, recursive=True)
    # print(img_path)
    if len(img_path) >= 1:
        img = skimage.io.imread(img_path[0])
    else:
        img = skimage.io.imread(file)
    shape, _ = my_shape_index(img)
    shape = np.nan_to_num(shape)
    hist, bins_shape = np.histogram(shape, bins=bins)
    hist = hist / np.max(hist)
    return hist


def edge_orientation_feature_extraction(file, subdir=PATH):
    img_path = glob.glob(subdir + "/**/" + file, recursive=True)
    # print(img_path)
    if len(img_path) >= 1:
        img = skimage.io.imread(img_path[0])
    else:
        img = skimage.io.imread(file)
    sy = sobel_h(img)
    sx = sobel_v(img)
    phi = np.arctan(np.divide(sy, sx))
    angles = np.nan_to_num(phi)
    hist, _ = np.histogram(angles, bins=360)
    hist = hist / np.max(hist)
    return hist


def cat_pca(feature_dict: dict) -> dict:
    pass


def binary_pca():
    pass


def eval_gmm(
    model: mixture.BayesianGaussianMixture, category: str, n_val_imgs: int, bins=9
):
    ll = 0
    # get validation data for category
    path = VAL_PATH
    # print(path)
    # files = os.listdir(path)
    files = glob.glob(path + "/**/*.png")
    files.extend(glob.glob(path + "/**/*.jpg"))
    # print(path)
    files = [i for i in files if re.findall("jpg", i) or re.findall("png", i)][
        0:n_val_imgs
    ]

    for file in files:
        hist = shape_feature_extraction(file, subdir=VAL_PATH, bins=bins)
        # hist = edge_orientation_feature_extraction(file, subdir=VAL_PATH)
        ll += model.score(hist.reshape(1, -1))

    return ll / len(files)


@ignore_warnings(category=ConvergenceWarning)
def build_gmm(features: list, category: str, search_space=15, n_val=1000, bins=9):
    max_ll = -np.inf
    best_n = 0
    for n_comp in range(1, search_space):
        model = mixture.BayesianGaussianMixture(
            n_components=n_comp, covariance_type="full"
        ).fit(features)

        ll = eval_gmm(model, category, n_val, bins=bins)
        # print(ll)
        if ll > max_ll:
            max_ll = ll
            best_n = n_comp

    best_model = mixture.BayesianGaussianMixture(
        n_components=best_n, covariance_type="full"
    ).fit(features)
    print(category, best_n)
    return best_model


def system_performance_eval(model_dict: dict, n_val: int, bins=9):
    accuracy_dict = {}
    categories = list(model_dict.keys())
    for category in categories:
        accuracy = 0
        # get val images
        path = VAL_PATH + category
        files = os.listdir(path)
        files = [i for i in files if re.findall("jpg", i) or re.findall("png", i)][
            0:n_val
        ]

        for file in files:
            scores = []
            # get feature
            hist = shape_feature_extraction(file, subdir=VAL_PATH, bins=bins)

            # score all samples
            for model_category in categories:
                scores.append(model_dict[model_category].score(hist.reshape(1, -1)))

            # compare scores, select category
            best_score_idx = scores.index(max(scores))
            best_category = categories[best_score_idx]
            if best_category == category:
                accuracy += 1

        accuracy_dict[category] = accuracy / n_val

    return accuracy_dict


@ignore_warnings(category=ConvergenceWarning)
def system_performance_eval_bin(
    model_dict: dict, n_val: int, imp_cats: list, meh_cats: list, bins=9
):
    accuracy_dict = {}
    categories = list(model_dict.keys())
    accuracy = 0

    # select n_val images from all interesting categories
    # then select same amount of images from uninteresting categories

    int_files = []
    for category in imp_cats:
        path = VAL_PATH + category
        files = os.listdir(path)[0 : int(np.floor(n_val / len(imp_cats)))]
        int_files.extend(files)
    unint_files = []
    for category in meh_cats:
        path = VAL_PATH + category
        files = os.listdir(path)[0 : int(np.floor(len(int_files) / len(meh_cats)))]
        unint_files.extend(files)

    for file in int_files:
        scores = []
        # get feature
        hist = shape_feature_extraction(file, subdir=VAL_PATH, bins=bins)
        # hist = edge_orientation_feature_extraction(file, subdir=VAL_PATH)

        # score all samples
        for model_category in categories:
            scores.append(model_dict[model_category].score(hist.reshape(1, -1)))

        # compare scores, select category
        best_score_idx = scores.index(max(scores))
        best_category = categories[best_score_idx]
        if best_category == "landmark":
            accuracy += 1

    accuracy_dict["landmark"] = accuracy / len(int_files)
    accuracy = 0
    for file in unint_files:
        scores = []
        # get feature
        hist = shape_feature_extraction(file, subdir=VAL_PATH, bins=bins)
        # hist = edge_orientation_feature_extraction(file, subdir=VAL_PATH)

        # score all samples
        for model_category in categories:
            scores.append(model_dict[model_category].score(hist.reshape(1, -1)))

        # compare scores, select category
        best_score_idx = scores.index(max(scores))
        best_category = categories[best_score_idx]
        if best_category == "terrain":
            accuracy += 1

    accuracy_dict["terrain"] = accuracy / len(unint_files)
    return accuracy_dict


def build_models_full(categories: list, feature_dict: dict) -> dict:
    model_dict = {}
    for category in categories:
        model_dict[category] = build_gmm(feature_dict[category], category)

    return model_dict


@ignore_warnings(category=ConvergenceWarning)
def build_models_bin(
    categories: list, imp_cats: list, feature_dict: dict, bins=9
) -> dict:
    model_dict = {}
    aggregated_features_imp = []
    aggregated_features_meh = []
    for category in categories:
        if category in imp_cats:
            aggregated_features_imp.extend(feature_dict[category])
        else:
            aggregated_features_meh.extend(feature_dict[category])

    model_dict["landmark"] = build_gmm(aggregated_features_imp, "landmark", bins=bins)
    model_dict["terrain"] = build_gmm(aggregated_features_meh, "terrain", bins=bins)
    return model_dict


if __name__ == "__main__":
    bins = range(4, 15)
    for curr_bin in bins:
        features = []
        feature_dict = {}
        categories = [
            "aec",
            "ael",
            "cli",
            "cra",
            "fse",
            "fsf",
            "fsg",
            "fss",
            "mix",
            "rid",
            "rou",
            "sfe",
            "sfx",
            "smo",
            "tex",
        ]
        imp_cat = [
            "aec",
            "ael",
            "cli",
            "cra",
            "fse",
            "fsf",
            "fsg",
            "fss",
            "sfe",
            "sfx",
            "rid",
        ]
        meh_cat = ["rou", "smo", "tex", "mix"]

        # select images to use
        for category in categories:
            path = PATH + category
            # print(path)
            files = os.listdir(path)
            # print(path)
            files = [i for i in files if re.findall("jpg", i) or re.findall("png", i)][
                :100
            ]

            # create features
            for file in files:
                shape_hist = shape_feature_extraction(file, bins=curr_bin)
                # shape_hist = edge_orientation_feature_extraction(file)
                if category not in feature_dict:
                    feature_dict[category] = []
                feature_dict[category].append(shape_hist)
                # TODO: quicksave here!
        print("Done extracting features")

        # TODO: test on individual landmarks, as well as binary categories

        # TODO: set up pca, first test without any, data dimensionality should be an issue
        # TODO: test on all data, but also try individual pcas for different classes

        # build up gmms
        model_dict = build_models_bin(categories, imp_cat, feature_dict, bins=curr_bin)
        # TODO: one for every category??
        # TODO: how do i eval the gmms? i WANT multiple components for every category

        # test gmm
        accuracies = system_performance_eval_bin(
            model_dict, n_val=1000, imp_cats=imp_cat, meh_cats=meh_cat, bins=curr_bin
        )
        print(accuracies)
        # TODO: select random samples (val set) for all categories, score sample, assign to
        # most likely gmm, calc accuracy
