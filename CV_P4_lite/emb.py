#!/usr/bin/env python
# coding: utf-8

from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os
import random

workers = 0 if os.name == 'nt' else 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

# helper function to align (if needed)
mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, prewhiten=True,
    device=device
)

#pretrained network 
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def collate_fn(x):
    return x[0]

dataset = datasets.ImageFolder('./data')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
print(dataset)
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)
print(loader)

aligned = []
names = []
idx = 0
for x, y in loader:
    # x_aligned, prob = mtcnn(x, return_prob=True)
    idx += 1
    print(idx)
    # if idx > 10: break
    if x is not None:
        # print('Face detected with probability: {:8f}'.format(prob))
        # aligned.append(x_aligned)
        temp = np.rollaxis(np.uint8(x)/255,2,0)
        # print(temp.shape)
        aligned.append(torch.tensor(temp).float())
        names.append(dataset.idx_to_class[y])
print("aligned",len(aligned))

embeddings = []


for i in range(1,53):
    print("B:",i,"---",(i-1)*50,"-",i*50-1)
    temp_aligned = torch.stack(aligned[(i-1)*50:i*50]).to(device)
    print(temp_aligned.shape)
    # temp_aligned = 
    # print(len(temp_aligned))
    embeddings.extend(resnet(temp_aligned).detach().cpu().numpy())
    print(len(embeddings))
    
print("len e",len(embeddings))
print("len e",len(embeddings[0]))
np.array(embeddings).dump(open('embedimg.npy', 'wb'))
np.array(names).dump(open('embedname.npy', 'wb'))



