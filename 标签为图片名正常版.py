import torch
from torch.utils.data import DataLoader,Dataset
import torchvision
from torchvision import datasets,utils,transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import PIL.Image
import os
import warnings
warnings.filterwarnings('ignore')
'''
提供了一种将图片和标签输出的方法
'''
transform = transforms.Compose([transforms.Resize(32),
                                transforms.CenterCrop(32),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.492,0.461,0.417],std=[0.256,0.248,0.251])
])
def show_imag(img):
    img = img/2+0.5
    npimg = img.numpy()
    npimg = npimg.reshape(-1, 32, 32)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # transpose函数用于交换维度
    npimg = np.transpose(npimg,(1,2,0))
    plt.imshow(npimg)
    plt.show()

class MyDataset(Dataset):
    def __init__(self,path_dir,transform=None):
        self.path = path_dir
        self.transforms = transform
        self.images_name = os.listdir(self.path)
    def __len__(self):
        return len(self.images_name)
    def __getitem__(self,index):
        index_name = self.images_name[index]
        index_path = os.path.join(self.path,index_name)
        img = PIL.Image.open(index_path).convert('RGB')

        label = index_path.split('\\')[-1].split('.')[0]
        label =1 if 'dog' in label else 0
        if self.transforms is not None:
            img = self.transforms(img)
        return img,label
imges,labels = MyDataset(r'D:\PyCharm\LeNet\dataset_kaggledogvscat\dataset_kaggledogvscat\train\train',transform)
dataloader = DataLoader(imges,batch_size=8,shuffle=True) #这里输出的图片为（batch_size,通道数，宽，高），不能直接输出图片

