import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms,datasets,utils
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

batch_size = 8
root = r'D:\PyCharm\数据增强\dogs_cats\dogs_cats\data\train'
trans = transforms.Compose([
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
train_data = datasets.ImageFolder(root=root,transform=trans)
train_iter = torch.utils.data.DataLoader(train_data,batch_size=batch_size,shuffle=True)
def im_show(img,ax):
    img = img.numpy()
    ax.imshow(np.transpose(img,(1,2,0)))

#显示图像增强前后图片对比
def img_trans_show(train_iter,transform):
    imgs,labels = next(iter(train_iter))
    img_initial = torchvision.utils.make_grid(imgs)
    img_trans = transform(img_initial)
    img_trans = torchvision.utils.make_grid(img_trans)
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)
    im_show(img_initial,ax1)
    im_show(img_trans,ax2)
    plt.show()
img_trans_show(train_iter=train_iter,transform=transforms.RandomVerticalFlip())
