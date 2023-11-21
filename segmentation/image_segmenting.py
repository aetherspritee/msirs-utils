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

# これを読む人はバカだよ
# Put on GPU if available
model = model.to(device)

# Set model to eval mode (turns off dropout and moving averages of batchnorm)
model.eval()
# path = f"/home/{USER}/codebase-v1/POIs/extracted_pois_ctx_crater.jpeg.png"
for file in file_names:
    file_name = file.split("/")[-1][:-4]
    step_size = 1
    plt.imsave(
        "segmented/" + file_name + "_" + network_name + str(step_size) + "_og_img.png",
        skimage.io.imread(file)
        )

    ctx_image = HIRISE_Image(path=file, transform=data_transform, step_size=step_size)

    test_loader = DataLoader(
        ctx_image, batch_size=64, shuffle=False, num_workers=8, pin_memory=True
    )
    segment_image(test_loader, model, device, hyper_params, step_size=step_size, img_name=file_name)
    print("DONE WITH FILE")
