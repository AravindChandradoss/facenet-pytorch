#!/usr/bin/env python
# coding: utf-8

# # Face detection and recognition inference pipeline
# 
# The following example illustrates how to use the `facenet_pytorch` python package to perform face detection and recogition on an image dataset using an Inception Resnet V1 pretrained on the VGGFace2 dataset.
# 
# The following Pytorch methods are included:
# * Datasets
# * Dataloaders
# * GPU/CPU processing

# In[1]:


from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import numpy as np
import pandas as pd
import os

workers = 0 if os.name == 'nt' else 4


# #### Determine if an nvidia GPU is available

# In[2]:


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))


# #### Define MTCNN module
# 
# Default params shown for illustration, but not needed. Note that, since MTCNN is a collection of neural nets and other code, the device must be passed in the following way to enable copying of objects when needed internally.
# 
# See `help(MTCNN)` for more details.

# In[3]:


mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, prewhiten=True,
    device=device
)


# #### Define Inception Resnet V1 module
# 
# Set classify=True for pretrained classifier. For this example, we will use the model to output embeddings/CNN features. Note that for inference, it is important to set the model to `eval` mode.
# 
# See `help(InceptionResnetV1)` for more details.

# In[4]:


resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)


# #### Define a dataset and data loader
# 
# We add the `idx_to_class` attribute to the dataset to enable easy recoding of label indices to identity names later one.

# In[5]:


def collate_fn(x):
    return x[0]

dataset = datasets.ImageFolder('../data/test_images')
dataset.idx_to_class = {i:c for c, i in dataset.class_to_idx.items()}
loader = DataLoader(dataset, collate_fn=collate_fn, num_workers=workers)


# #### Perfom MTCNN facial detection
# 
# Iterate through the DataLoader object and detect faces and associated detection probabilities for each. The `MTCNN` forward method returns images cropped to the detected face, if a face was detected. By default only a single detected face is returned - to have `MTCNN` return all detected faces, set `keep_all=True` when creating the MTCNN object above.
# 
# To obtain bounding boxes rather than cropped face images, you can instead call the lower-level `mtcnn.detect()` function. See `help(mtcnn.detect)` for details.

# In[6]:


aligned = []
names = []
for x, y in loader:
    x_aligned, prob = mtcnn(x, return_prob=True)
    print(x_aligned.shape)
    if x_aligned is not None:
        print('Face detected with probability: {:8f}'.format(prob))
        aligned.append(x_aligned)
        names.append(dataset.idx_to_class[y])


# #### Calculate image embeddings
# 
# MTCNN will return images of faces all the same size, enabling easy batch processing with the Resnet recognition module. Here, since we only have a few images, we build a single batch and perform inference on it. 
# 
# For real datasets, code should be modified to control batch sizes being passed to the Resnet, particularly if being processed on a GPU. For repeated testing, it is best to separate face detection (using MTCNN) from embedding or classification (using InceptionResnetV1), as calculation of cropped faces or bounding boxes can then be performed a single time and detected faces saved for future use.

# In[7]:


aligned = torch.stack(aligned).to(device)
print("a",aligned.shape)
batch = aligned[:2,...]
print("b",batch.shape)
embeddings = resnet(batch).detach().cpu()
print("e",embeddings.shape)


# #### Print distance matrix for classes

# In[8]:


dists = [[(e1 - e2).norm().item() for e2 in embeddings] for e1 in embeddings]
print(dists)
# print(pd.DataFrame(dists, columns=names, index=names))

