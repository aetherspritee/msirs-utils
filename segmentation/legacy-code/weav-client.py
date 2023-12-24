import weaviate
import base64
import os
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib
import random
import re

# matplotlib.use("QtAgg",force=True)
USER = "dusc"


def add_imgs(client, imgs):
    for img in imgs:
        with open(img, "rb") as image:
            decoded_img = base64.b64encode(image.read())
            # print(type(decoded_img))
            client.data_object.create(
                {
                    "image": decoded_img.decode("utf-8"),
                },
                "SegmentedImg",
            )
    return client


client = weaviate.Client("http://localhost:8080")

# FIXME: To delete a class
# client.schema.delete_class("SegmentedImg")
# client.schema.delete_class("MarsImg")

schema = {
    "classes": [
        {
            "class": "SegmentedImg",
            "vectorizer": "img2vec-neural",
            "vectorIndexType": "hnsw",
            "moduleConfig": {"img2vec-neural": {"imageFields": ["image"]}},
            "properties": [{"name": "image", "dataType": ["blob"]}],
        }
    ]
}

client.schema.create(schema)

# print(client.schema.get())

# path = f"/home/{USER}/Dropbox/Stuff/Code/LSIR/MarsImgs/cropped_and_named/"
# imgs = list(os.listdir(path))
# imgs = [path+i for i in imgs]
# client = add_imgs(client, imgs)
# test_path = f"/home/{USER}/Dropbox/Stuff/Code/LSIR/MarsImgs/"
test_path = f"/Users/{USER}/segmentation/segmented/"
all_imgs = os.listdir(test_path)
all_imgs = [i for i in all_imgs if re.findall("mrf", i)]
# to_db_imgs = [test_path+i for i in all_imgs]
# client = add_imgs(client, to_db_imgs)
# print("Populated database..")
q_img = random.choice(all_imgs)
test = {"image": test_path + q_img}

# q_img = Image.open(test_path + q_img)
with open(test_path + q_img, "rb") as imgg:
    dec_img = base64.b64encode(imgg.read())
    # print(type(decoded_img))
    client.data_object.create(
        {
            "image": dec_img.decode("utf-8"),
        },
        "SegmentedImg",
    )
# res = client.get().withClassName("MarsImg").withFields(['image']).withNearImage({"image": test}).withLimit(10).do()
result = client.query.get("SegmentedImg", "image").do()
print(result)
res = [i["image"] for i in result["data"]["Get"]["SegmentedImg"]]
print(res)
# print("dooo")
img_raw = base64.b64decode(res[0])
with open("result1.jpg", "wb") as f:
    f.write(img_raw)
img_raw = base64.b64decode(res[-1])
with open("resultE.jpg", "wb") as f:
    f.write(img_raw)


plt.figure()
plt.subplot(1, 4, 1)
plt.imshow(q_img)
res = res[0:3]
for i in range(len(res)):
    plt.subplot(1, 4, i + 2)
    img_raw = base64.b64decode(res[i])
    with open("resultQ.jpg", "wb") as f:
        f.write(img_raw)
    r_img = Image.open("resultQ.jpg")
    plt.imshow(r_img)
plt.show()

print("WOOOO")
