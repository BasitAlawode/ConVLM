
import scipy.io as sio
import torch
from torch import nn
import torchvision
from torchvision import transforms, datasets
import torchvision.transforms.functional as TF
import torchvision.models as models
from torch.utils.data import DataLoader,TensorDataset
import PIL
import os
from matplotlib import pyplot as plt
import numpy as np
import itertools
import pandas as  pd
from uni import get_encoder
from uni.downstream.extract_patch_features import extract_patch_features_from_dataloader
import timm
from huggingface_hub import login, hf_hub_download


torch.cuda.empty_cache()

def dir_creat(path):
    if not os.path.exists(path):
        os.mkdir(path)


login()  # login with your User Access Token, found at https://huggingface.co/settings/tokens

local_dir = "/content/drive/MyDrive/ConVLM_model/vit_large_patch16_224.dinov2.uni_mass100k/"
os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=True)
model = timm.create_model(
    "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
)

device = 'cuda'
model.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
model.eval()
model.to(device)
transform = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


model, transform = get_encoder(enc_name='uni', device=device)


# get path to data
dataroot = '/content/drive/MyDrive/ConVLM_data/NSCLC_TCGA'

# create some image folder datasets for train/test and their data laoders
train_dataset = torchvision.datasets.ImageFolder(os.path.join(dataroot, 'train'), transform=transform)
test_dataset = torchvision.datasets.ImageFolder(os.path.join(dataroot, 'test'), transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False)

# extract patch features from the train and test datasets (returns dictionary of embeddings and labels)
train_features = extract_patch_features_from_dataloader(model, train_dataloader)
test_features = extract_patch_features_from_dataloader(model, test_dataloader)

# convert these to torch
train_feats = torch.Tensor(train_features['embeddings'])
train_labels = torch.Tensor(train_features['labels']).type(torch.long)
test_feats = torch.Tensor(test_features['embeddings'])
test_labels = torch.Tensor(test_features['labels']).type(torch.long)


