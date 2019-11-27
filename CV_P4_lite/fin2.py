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
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
            discriminant_analysis, random_projection)
from sklearn.metrics import confusion_matrix

workers = 0 if os.name == 'nt' else 4

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

embed = np.load(open('embedimg.npy','rb'))
names = np.load(open('embedname.npy','rb'))

# =====================
# helper functions for plotting
# from sklearn library and stackoverflow
# =====================
import matplotlib.pyplot as plt
def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    #plt.tight_layout()
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    plt.show()
def plot_embedding(X, title=None):
   x_min, x_max = np.min(X, 0), np.max(X, 0)
   X = (X - x_min) / (x_max - x_min)     
   plt.figure()
   ax = plt.subplot(111)
   for i in range(X.shape[0]):
       plt.text(X[i, 0], X[i, 1], str(names[i]),
            color=plt.cm.Set1(int(names[i]) / 200.),
                fontdict={"weight": "bold", "size": 9})
   plt.xticks([]), plt.yticks([])
   if title is not None:
       plt.title(title)
       plt.show()

# # =====================
# #     MAIN CODE
# # one shot learning
# # using leave out one image 
# # =====================
conf_a = []
conf_p = []
for i in range(1,101):
    for _ in range(10):
        idx = random.randint((i-1)*26,i*26-1) # any img
        face1 = embed[idx]
        mini = float("inf")
        predic = -1
        for face2 in range(len(embed)): # any other imf (except the chossen)
            if face2 == idx: 
                continue
            error = torch.from_numpy((face1 - embed[face2])).to(device).norm().item()
            if error < mini:
                mini = error
                predic = face2
        print("Idx:",idx, predic,"Actual:",names[idx],"pred:",names[predic], "error:",error)
        conf_a.append(names[idx])
        conf_p.append(names[predic])

y_actu = pd.Series(conf_a, name='Actual')
y_pred = pd.Series(conf_p, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred) 
print(df_confusion)
plot_confusion_matrix(df_confusion)

# # =====================
# # using one image as identity to compare
# # =====================

conf_a = []
conf_p = []
correct = 0
incorrect = 0
for i in range(1,101):
    for _ in range(10):
        idx = random.randint((i-1)*26,i*26-1) #any image
        face1 = embed[idx]
        mini = float("inf")
        predic = -1
        for face2 in range(0,2600,13): # using only the first image (as model)
            if face2 == idx: 
                continue
            error = torch.from_numpy((face1 - embed[face2])).to(device).norm().item()
            if error < mini:
                mini = error
                predic = face2
        print("Idx:",idx, predic,"Actual:",names[idx],"pred:",names[predic], "error:",error)
        conf_a.append(names[idx])
        conf_p.append(names[predic])
        if names[idx] == names[predic]:
            correct += 1
        else: 
            incorrect += 1

y_actu = pd.Series(conf_a, name='Actual')
y_pred = pd.Series(conf_p, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred) 
print(df_confusion)
plot_confusion_matrix(df_confusion)
print("correct:", correct, "incorrect:", incorrect, "=",correct/(correct+incorrect),"%")


# # =====================
# # using one image as identity to compare
# # For session
# # =====================
conf_a = []
conf_p = []
correct = 0
incorrect = 0
for i in range(1,101):
    for _ in range(10):
        idx = random.randint((i-1)*26+13,i*26-1) # choosing from 2nd sess
        face1 = embed[idx]
        mini = float("inf")
        predic = -1
        for face2 in range(0,2600):
            if face2 == idx or face2%26 > 13: # using any img from 1st sess
                continue
            error = torch.from_numpy((face1 - embed[face2])).to(device).norm().item()
            if error < mini:
                mini = error
                predic = face2
        print("Idx:",idx, predic,"Actual:",names[idx],"pred:",names[predic], "error:",error)
        conf_a.append(names[idx])
        conf_p.append(names[predic])
        if names[idx] == names[predic]:
            correct += 1
        else: 
            incorrect += 1

y_actu = pd.Series(conf_a, name='Actual')
y_pred = pd.Series(conf_p, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred) 
print(df_confusion)
plot_confusion_matrix(df_confusion)
print("correct:", correct, "incorrect:", incorrect, "=",correct/(correct+incorrect),"%")


# # =====================
# # using one image as identity to compare
# # For session
# # =====================
conf_a = []
conf_p = []
correct = 0
incorrect = 0
for i in range(1,101):
    for _ in range(10):
        idx = random.randint((i-1)*26+13,i*26-1) # choosing img from 2nd sess
        face1 = embed[idx]
        mini = float("inf")
        predic = -1
        for face2 in range(0,2600,26): #using only one image (from 1st session)
            if face2 == idx: 
                continue
            error = torch.from_numpy((face1 - embed[face2])).to(device).norm().item()
            if error < mini:
                mini = error
                predic = face2
        print("Idx:",idx, predic,"Actual:",names[idx],"pred:",names[predic], "error:",error)
        conf_a.append(names[idx])
        conf_p.append(names[predic])
        if names[idx] == names[predic]:
            correct += 1
        else: 
            incorrect += 1

y_actu = pd.Series(conf_a, name='Actual')
y_pred = pd.Series(conf_p, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred) 
print(df_confusion)
plot_confusion_matrix(df_confusion)
print("correct:", correct, "incorrect:", incorrect, "=",correct/(correct+incorrect),"%")


# # =====================
# # using pca model 
# # =====================
data = np.array(embed)
model_mean = []
model_vector = []
model_label = []
for i in range(0,100):
    idx = list(range((i-1)*26,i*26-13))
    x = data[idx]
    tsum = np.mean(x,axis=0)
    x = np.subtract(x,tsum)
    xtx = np.transpose(x)@x 
    u, s, v = np.linalg.svd(x, full_matrices=True)
    model_mean.append(tsum)
    model_vector.append(v)
    model_label.append(names[idx[0]])
print(len(model_mean))


# # =====================
# # using All Eigen vectors
# # =====================

conf_a = []
conf_p = []
correct = 0
incorrect = 0
for i in range(1,101):
    for _ in range(10):
        idx = random.randint((i-1)*26+13,i*26-1) # choosing from 2nd sess
        face1 = embed[idx]
        mini = float("inf")
        predic = -1
        for model, (mean, vector, label) in enumerate(zip(model_mean, model_vector, model_label)):
            error = face1 - (vector@(np.transpose(vector)@(face1-mean)) + mean)
            error = np.linalg.norm(error)
            if error < mini:
                mini = error
                predic = label
        print("Idx:",idx, predic,"Actual:",names[idx],"pred:",predic, "error:",error)
        conf_a.append(names[idx])
        conf_p.append(predic)
        if names[idx] == predic:
            correct += 1
        else: 
            incorrect += 1

y_actu = pd.Series(conf_a, name='Actual')
y_pred = pd.Series(conf_p, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred) 
print(df_confusion)
plot_confusion_matrix(df_confusion)
print("correct:", correct, "incorrect:", incorrect, "=",correct/(correct+incorrect),"%")


# # =====================
# # using 10 Eigen vectors
# # =====================

conf_a = []
conf_p = []
correct = 0
incorrect = 0
for i in range(1,101):
    for _ in range(10):
        idx = random.randint((i-1)*26+13,i*26-1) # choosing from 2nd sess
        face1 = embed[idx]
        mini = float("inf")
        predic = -1
        for model, (mean, vector, label) in enumerate(zip(model_mean, model_vector, model_label)):
            error = face1 - (vector[:,1:10]@(np.transpose(vector[:,1:10])@(face1-mean)) + mean)
            error = np.linalg.norm(error)
            if error < mini:
                mini = error
                predic = label
        print("Idx:",idx, predic,"Actual:",names[idx],"pred:",predic, "error:",error)
        conf_a.append(names[idx])
        conf_p.append(predic)
        if names[idx] == predic:
            correct += 1
        else: 
            incorrect += 1

y_actu = pd.Series(conf_a, name='Actual')
y_pred = pd.Series(conf_p, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred) 
print(df_confusion)
plot_confusion_matrix(df_confusion)
print("correct:", correct, "incorrect:", incorrect, "=",correct/(correct+incorrect),"%")


# # =====================
# # using 5 Eigen vectors
# # =====================

conf_a = []
conf_p = []
correct = 0
incorrect = 0
for i in range(1,101):
    for _ in range(10):
        idx = random.randint((i-1)*26+13,i*26-1) # choosing from 2nd sess
        face1 = embed[idx]
        mini = float("inf")
        predic = -1
        for model, (mean, vector, label) in enumerate(zip(model_mean, model_vector, model_label)):
            error = face1 - (vector[:,1:5]@(np.transpose(vector[:,1:5])@(face1-mean)) + mean)
            error = np.linalg.norm(error)
            if error < mini:
                mini = error
                predic = label
        print("Idx:",idx, predic,"Actual:",names[idx],"pred:",predic, "error:",error)
        conf_a.append(names[idx])
        conf_p.append(predic)
        if names[idx] == predic:
            correct += 1
        else: 
            incorrect += 1

y_actu = pd.Series(conf_a, name='Actual')
y_pred = pd.Series(conf_p, name='Predicted')
df_confusion = pd.crosstab(y_actu, y_pred) 
print(df_confusion)
plot_confusion_matrix(df_confusion)
print("correct:", correct, "incorrect:", incorrect, "=",correct/(correct+incorrect),"%")



# =====================
# visualizing eigen bases
# =====================

X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(embed)
plot_embedding(X_pca,"PCA")

tsne = manifold.TSNE(n_components=2, init="pca", random_state=0)
X_tsne = tsne.fit_transform(embed)
plot_embedding(X_tsne,"“t-SNE plot")

