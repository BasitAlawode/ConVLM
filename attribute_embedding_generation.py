import torch
import numpy as np
import clip
import os 
import scipy.io as sio
import torch
from sklearn import preprocessing
from torch.utils.data import Dataset
from torch import Tensor
import torch.nn as nn


############### Class name vocabulary ##############

model.load_state_dict(checkpoint)

class_name= ["Lung Adenocarcinoma", "Lung Squamous Cell Carcinoma"]
vocab = []
for n in class_name:
    vocab.append(n)

################## QuiltNet Embedding for attribute #############
device = "cpu" #"cuda" if torch.cuda.is_available() else "cpu"

quilt_model, preprocess = clip.create_model_from_pretrained('hf-hub:wisdomik/QuiltNet-B-32')
tokenizer = clip.get_tokenizer('hf-hub:wisdomik/QuiltNet-B-32')

input_text =["This histopathology image likely shows lung cancer tissue characterized by irregular, dense cluster of malignant cells”, “This histopathology image likely shows Lung Squamous Cell Carcinoma, characterized by malignant squamous cells with features such as keratinization and intercellular bridges, typically originating from the central airways."]

text = tokenizer(input_text).to(device)

with torch.no_grad():
    text_features = quilt_model(text)
    embeddings = text_features / torch.norm(text_features.float(), dim=-1).unsqueeze(-1)




