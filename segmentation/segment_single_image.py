#!/usr/bin/env python3

import numpy as np
import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import glob
from domars_map import MarsModel, HIRISE_Image, segment_image
from PIL import Image
import matplotlib as mpl
from matplotlib import pyplot as plt

network_name = "densenet161"
USER = "pg2022"
torch.multiprocessing.set_sharing_strategy("file_system")

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
    "model": network_name,
    "num_classes": 15,
    "pretrained": False,
    "transfer_learning": False,
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model = MarsModel(hyper_params)
print(f"/home/{USER}/models")
checkpoint = torch.load(
    f"/home/{USER}/models/" + network_name + ".pth", map_location=torch.device("cpu")
)
model.load_state_dict(checkpoint)


model = model.to(device)

# Set model to eval mode (turns off dropout and moving averages of batchnorm)
model.eval()

image_path = "/home/pg2022/codebase-v1/data/data/train/mix/D09_030608_1812_XI_01N359W_CX9768_CY37783.jpg"

#ctx_image = HIRISE_Image(path=image_path, transform=data_transform, step_size=2)

ctx_image = data_transform(Image.open(image_path).convert('RGB'))

test_img1 = ctx_image.unsqueeze(0)
vec_rep = model(test_img1.to(device))
pred = torch.argmax(vec_rep, dim=1).cpu()

img_name = "test_mix"

new_img = np.ones((200,200))*int(pred)
n = int(hyper_params["num_classes"])
from_list = mpl.colors.LinearSegmentedColormap.from_list
cm = from_list(None, plt.cm.tab20(range(0, n)), n)
plt.imsave(
    "segmented/" + img_name + "_" + network_name + "_map.png",
    new_img,
    cmap=cm,
    vmin=0,
    vmax=int(hyper_params["num_classes"]),
)

