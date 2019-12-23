# Face Recognition Using One Shot Learning (Siamese network) and Model based (PCA) with FaceNet_Pytorch 

Refer to the [report](https://github.com/AravindChandradoss/facenet-pytorch/blob/master/CV_P4/Report.pdf) for results! 
https://aravindchandradoss.github.io/facenet-pytorch/



| System | Python | |
| :---: | :---: | :---: |
| Linux | 3.5, 3.6, 3.7 | [![Build Status](https://travis-ci.com/timesler/facenet-pytorch.svg?branch=master)](https://travis-ci.com/timesler/facenet-pytorch) |
| macOS | 3.6, 3.7 | [![Build Status](https://travis-ci.com/timesler/facenet-pytorch.svg?branch=master)](https://travis-ci.com/timesler/facenet-pytorch) |
| Windows | 3.5, 3.6, 3.7 | [![Build Status](https://travis-ci.com/timesler/facenet-pytorch.svg?branch=master)](https://travis-ci.com/timesler/facenet-pytorch) |

This is a repository for Inception Resnet (V1) models in pytorch, pretrained on VGGFace2 and CASIA-Webface.

Pytorch model weights were initialized using parameters ported from David Sandberg's [tensorflow facenet repo](https://github.com/davidsandberg/facenet).

Also included in this repo is an efficient pytorch implementation of MTCNN for face detection prior to inference. These models are also pretrained.

### This repo is build on top of [facenet-pytorch](https://github.com/timesler/facenet-pytorch)

## Quick start

1. Either install using pip:
    ```bash
    pip install facenet-pytorch
    ```
    or clone this repo, removing the '-' to allow python imports:
    ```bash
    git clone https://github.com/timesler/facenet-pytorch.git facenet_pytorch
    ```
    or use a docker container (see [timesler/jupyter-dl-gpu](https://github.com/timesler/docker-jupyter-dl-gpu)):
    ```bash
    docker run -it --rm timesler/jupyter-dl-gpu pip install facenet-pytorch && ipython
    ```
1. In python, import the module:
    ```python
    from facenet_pytorch import MTCNN, InceptionResnetV1
    ```
1. If required, create a face _detection_ pipeline using MTCNN:
    ```python
    mtcnn = MTCNN(image_size=<image_size>, margin=<margin>)
    ```
1. Create an inception resnet (in eval mode):
    ```python
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    ```
1. Process an image:
    ```python
    from PIL import Image
    
    img = Image.open(<image path>)

    # Get cropped and prewhitened image tensor
    img_cropped = mtcnn(img, save_path=<optional save path>)

    # Calculate embedding (unsqueeze to add batch dimension)
    img_embedding = resnet(img_cropped.unsqueeze(0))

    # Or, if using for VGGFace2 classification
    resnet.classify = True
    img_probs = resnet(img_cropped.unsqueeze(0))
    ```

See `help(MTCNN)` and `help(InceptionResnetV1)` for usage and implementation details.

## Pretrained models

See: [models/inception_resnet_v1.py](models/inception_resnet_v1.py)

The following models have been ported to pytorch (with links to download pytorch state_dict's):

|Model name|LFW accuracy (as listed [here](https://github.com/davidsandberg/facenet))|Training dataset|
| :- | :-: | -: |
|[20180408-102900](https://drive.google.com/uc?export=download&id=12DYdlLesBl3Kk51EtJsyPS8qA7fErWDX) (111MB)|0.9905|CASIA-Webface|
|[20180402-114759](https://drive.google.com/uc?export=download&id=1TDZVEBudGaEd5POR5X4ZsMvdsh1h68T1) (107MB)|0.9965|VGGFace2|

There is no need to manually download the pretrained state_dict's; they are downloaded automatically on model instantiation and cached for future use in the torch cache. To use an Inception Resnet (V1) model for facial recognition/identification in pytorch, use:

```python
from facenet_pytorch import InceptionResnetV1

# For a model pretrained on VGGFace2
model = InceptionResnetV1(pretrained='vggface2').eval()

# For a model pretrained on CASIA-Webface
model = InceptionResnetV1(pretrained='casia-webface').eval()

# For an untrained model with 100 classes
model = InceptionResnetV1(num_classes=100).eval()

# For an untrained 1001-class classifier
model = InceptionResnetV1(classify=True, num_classes=1001).eval()
```

Both pretrained models were trained on 160x160 px images, so will perform best if applied to images resized to this shape. For best results, images should also be cropped to the face using MTCNN (see below).

By default, the above models will return 512-dimensional embeddings of images. To enable classification instead, either pass `classify=True` to the model constructor, or you can set the object attribute afterwards with `model.classify = True`. For VGGFace2, the pretrained model will output logit vectors of length 8631, and for CASIA-Webface logit vectors of length 10575.

## References

1. David Sandberg's facenet repo: [https://github.com/davidsandberg/facenet](https://github.com/davidsandberg/facenet)

1. F. Schroff, D. Kalenichenko, J. Philbin. _FaceNet: A Unified Embedding for Face Recognition and Clustering_, arXiv:1503.03832, 2015. [PDF](https://arxiv.org/pdf/1503.03832)

1. Q. Cao, L. Shen, W. Xie, O. M. Parkhi, A. Zisserman. _VGGFace2: A dataset for recognising face across pose and age_, International Conference on Automatic Face and Gesture Recognition, 2018. [PDF](http://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/cao18.pdf)

1. D. Yi, Z. Lei, S. Liao and S. Z. Li. _CASIAWebface: Learning Face Representation from Scratch_, arXiv:1411.7923, 2014. [PDF](https://arxiv.org/pdf/1411.7923)

1. K. Zhang, Z. Zhang, Z. Li and Y. Qiao. _Joint Face Detection and Alignment Using Multitask Cascaded Convolutional Networks_, IEEE Signal Processing Letters, 2016. [PDF](https://kpzhang93.github.io/MTCNN_face_detection_alignment/paper/spl.pdf)
