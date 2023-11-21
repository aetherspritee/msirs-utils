#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
from six import BytesIO

import tensorflow as tf

import tensorflow_hub as hub
from six.moves.urllib.request import urlopen


def run_delf(image):
  np_image = np.array(image)
  float_image = tf.image.convert_image_dtype(np_image, tf.float32)

  return delf(
      image=float_image,
      score_threshold=tf.constant(100.0),
      image_scales=tf.constant([0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]),
      max_feature_num=tf.constant(1000))

def match_images(image1, image2, result1, result2):
    distance_threshold = 0.8

    # Read features.
    num_features_1 = result1['locations'].shape[0]
    print("Loaded image 1's %d features" % num_features_1)

    num_features_2 = result2['locations'].shape[0]
    print("Loaded image 2's %d features" % num_features_2)

    implot = plt.imshow(image1)
    print(result1['locations'].numpy())
    plt.scatter(result1['locations'].numpy()[:,1],result1['locations'].numpy()[:,0])

    plt.show()

    implot = plt.imshow(image2)
    print(result1['locations'].numpy())
    plt.scatter(result2['locations'].numpy()[:,1],result2['locations'].numpy()[:,0])

    plt.show()


def show_descriptors(image1, result1):
    distance_threshold = 0.8

    # Read features.
    num_features_1 = result1['locations'].shape[0]
    print("Loaded image 1's %d features" % num_features_1)


    implot = plt.imshow(image1)
    print(result1['locations'].numpy())
    plt.scatter(result1['locations'].numpy()[:,0],result1['locations'].numpy()[:,1])

    plt.show()

if __name__ == "__main__":
    new_width=256
    new_height=256
    img1_path = "/Users/dusc/Dropbox/Stuff/Code/LSIR/data/data/train/cra/D09_030608_1812_XI_01N359W_CX4180_CY8517.jpg"
    img1_path = "../test_strip_full.png"
    img1 = Image.open(img1_path)
    image1 = ImageOps.fit(img1, (new_width,new_height), Image.ANTIALIAS)
    image1 = Image.fromarray(np.array(image1)[:,:,:3])
    print(np.shape(image1))
    # img2_path = "/Users/dusc/Dropbox/Stuff/Code/LSIR/data/data/train/cra/K01_053719_1938_XI_13N232W_CX3595_CY8247.jpg"
    # img2 = Image.open(img2_path)
    # image2 = ImageOps.fit(img2, (new_width,new_height), Image.ANTIALIAS)
    # image1 = image1.convert('RGB')
    # image2 = image2.convert('RGB')

    # plt.subplot(1,2,1)
    # plt.imshow(image1)
    # plt.subplot(1,2,2)
    # plt.imshow(image2)
    # plt.show()

    delf = hub.load('https://tfhub.dev/google/delf/1').signatures['default']
    print("Loaded network")
    # image1 = tf.expand_dims(image1,0)
    # image2 = tf.expand_dims(image2,0)

    result1 = run_delf(image1)

    show_descriptors(image1, result1)
