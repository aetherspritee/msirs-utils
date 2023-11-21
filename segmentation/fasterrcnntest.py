#!/usr/bin/env python3

import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import glob
from domars_map import MarsModel, HIRISE_Image, segment_image
from matplotlib import pyplot as plt
import skimage.io

USER = "pg2022"
network_name = "densenet161"
database_directory = f"/home/{USER}/mars-api/database"
print(f"{database_directory}")
file_names = glob.glob(database_directory + "/**/*.jpg", recursive=True)
print(os.listdir(database_directory))
print(file_names)

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
