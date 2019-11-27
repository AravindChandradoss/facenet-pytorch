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
    x_aligned, prob = mtcnn(x, return_prob=True)
    # idx += 1
    # if idx > 10: break
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])
print("aligned",len(aligned))

print(len(aligned))
embeddings = []

# for i in range(1,10):
# 	temp_aligned = torch.stack(aligned).to(device)
# 	embeddings.extend(resnet(temp_aligned).detach().cpu())


# for i in range(1,27):
# 	temp_aligned = torch.stack(aligned[(i-1)*100:i*100-1],...).to(device)
# 	embeddings.extend(resnet(temp_aligned).detach().cpu())


# print("len e",len(embeddings))
# print("len e",len(embeddings[0]))
# print(embeddings[0])
# np.array(embeddings).dump(open('array.npy', 'wb'))

# one shot learning
# for i in range(101):
# 	# idx = random.randint((i-1)*26,i*26-1)
# 	idx = random.randint(1,10)
# 	f1 = embeddings[idx]
# 	mini = float("inf")
# 	predic = -1
# 	for ii in range(len(embeddings)):
# 		if ii == idx: continue
# 		error = (f1 - embeddings[ii]).norm().item()
# 		if error < mini:
# 			mini = error
# 			predic = ii%26
# 	print("Idx:",idx,"pred:",predic)




# embeddings[(i-1)*26:i*26-1]



# # #### Print distance matrix for classes

# # In[8]:


# dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
# print(pd.DataFrame(dists, columns=names, index=names))




