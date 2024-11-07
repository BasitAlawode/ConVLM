import numpy as np
import os 
import torch, torchvision
import torch.nn as nn
from torch import Tensor
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
#from ctran import ctranspath
#import open_clip
from open_clip import create_model_from_pretrained, get_tokenizer # works on open-clip-torch>=2.23.0, timm>=0.9.8
import timm


############### Class name vocabulary ##############

class_name= ["Lung Adenocarcinoma", "Lung Squamous Cell Carcinoma"]
vocab = []
for n in class_name:
    vocab.append(n)

################## QuiltNet Embedding for attribute #############
device = "cuda" if torch.cuda.is_available() else "cpu"

quilt_model, preprocess = create_model_from_pretrained('hf-hub:wisdomik/QuiltNet-B-32')
tokenizer = get_tokenizer('hf-hub:wisdomik/QuiltNet-B-32')

input_text =["This histopathology image likely shows lung cancer tissue characterized by irregular, dense cluster of malignant cells”, “This histopathology image likely shows Lung Squamous Cell Carcinoma, characterized by malignant squamous cells with features such as keratinization and intercellular bridges, typically originating from the central airways."]

text = tokenizer(input_text).to(device)

with torch.no_grad():
    text_features = quilt_model.encode_text(text)(text)
    embeddings = text_features / torch.norm(text_features.float(), dim=-1).unsqueeze(-1)




