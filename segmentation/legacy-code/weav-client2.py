#!/usr/bin/env python3

import weaviate
import base64
import os
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
import random
import re
import skimage.io
from process_image import post_process

USER = "dusc"
client = weaviate.Client("http://localhost:8080")

def create_db(client,imgs, class_name):
    for img in imgs:
        info = post_process(img)
        info_l = [info[i] for i in info.keys()]
        data = {"name": img}
        client.data_object.create(data, class_name, vector=info_l)
    return client

def query_img(client, img, class_name):
    info = post_process(img)
    info_l = {"vector": [info[i] for i in info.keys()]}
    result = (
        client.query
        .get(class_name, ["name"])
        .with_near_vector(info_l)
        .with_additional(["distance"])
        .with_limit(5)
        .do()
    )
    res = [i["name"] for i in result["data"]["Get"][class_name]]
    return res

def plot_results(query,res):
    for i in range(len(res)):
        img_raw = skimage.io.imread(res[i])[:,:,:3]
        skimage.io.imsave(f"res{i}.jpg", img_raw)

    q_img = skimage.io.imread(query)[:,:,:3]
    plt.figure()
    plt.subplot(1,6,1)
    plt.imshow(q_img)
    for i in range(len(res)):
        plt.subplot(1,6,i+2)
        r_img = skimage.io.imread(f"res{i}.jpg")
        plt.imshow(r_img)
    plt.show()

schema = {
    'classes': [ {
        'class': 'SegmentedImg',
        'vectorizer': 'none',
        }]
    }

schema2 = {
    'classes': [ {
        'class': 'RegionDesc',
        'vectorizer': 'img2vec-neural',
        'vectorIndexType': 'hnsw',
        'moduleConfig': {
            '': {
                'imageFields': [
                    'name'
                    ]
                }
            },
        'properties': [
            {
            'name': 'image',
            'dataType': ['blob']
            }
            ]
        }]
    }



# client.schema.delete_class("test1")
client.schema.create(schema)

# print(client.schema.get())
#some_objects = client.data_object.get()
# print(some_objects)

#test_path = f"/Users/{USER}/segmentation/segmented/"
#all_imgs = os.listdir(test_path)
#all_imgs = [i for i in all_imgs if re.findall("mrf", i)]
#all_imgs = [test_path+i for i in all_imgs]

#img = test_path+random.choice(os.listdir(test_path))
#where_filter = {
#    "path": ["name"],
#    "operator": "Equal",
#    "valueText": img,
#}
#
#result = (
#    client.query
#    .aggregate("SegmentedImg")
#    .with_fields("meta {count}")
#    .with_where(where_filter)
#    .do()
#)
#print(result)
# client = create_db(client, all_imgs, "SegmentedImg")
# img = test_path+random.choice(os.listdir(test_path))
# res = query_img(client, img, "SegmentedImg")
# plot_results(img, res)
