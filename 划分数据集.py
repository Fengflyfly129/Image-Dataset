import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms as T
from torchvision import transforms,datasets,utils
import numpy as np
import pandas as pd
from PIL import Image
#import torch.nn.functional as F
#import torch.optim as optim
import os
import warnings
warnings.filterwarnings('ignore')

class MyDataset(Dataset):
    def __init__(self,root,transform=None,train=True,test=False): #初始化并没有什么要求，有的返回数据集的图像的名字，有的返回数据集图像的绝对地址
        self.test = test
        self.transforms = transform
        imgs = [os.path.join(root,img) for img in os.listdir(root)]
        if self.test:#划分测试集
            imgs = sorted(imgs,key = lambda x:int(x.split('.')[-2].split('\\')[-1])) #sorted默认升序排列，key为排序规则，这句代码为提取名字中的数字
            #对测试集的数据进行排序
        else:
            imgs = sorted(imgs,key = lambda x:int(x.split('.')[-2]))#取标签数字值，
            #排序的目的是便于后续的分割（为什么要排序）
        imgs_num = len(imgs)
        if self.test:#测试集
            self.imgs = imgs#测试集数据无需划分，直接导入
        elif train:
            self.imgs = imgs[:int(0.7*imgs_num)]#训练集
        else :
            self.imgs = imgs[int(0.7*imgs_num):]#验证集
        if self.transforms is None:
            normalize = T.Normalize(mean=[0.492,0.461,0.417],std=[0.256,0.248,0.251])
            if self.test or not train:#测试集和验证集
                self.transforms = T.Compose([
                    T.Resize(28),
                    T.CenterCrop(28),
                    T.ToTensor(),
                    normalize
                ])
            else:#训练集
                self.transforms = T.Compose([
                    T.Resize(28),
                    T.CenterCrop(28),
                    T.RandomHorizontalFlip(),
                    T.ToTensor(),
                    normalize
                ])
    def __len__(self):#返回数据集大小
        return len(self.imgs)
    def __getitem__(self, index):#根据索引index返回图像及标签
        img_path = self.imgs[index]
        if self.test:#测试集没有标签，随意取的，
            label = int(img_path.split('.')[-2].split('\\')[-1])
            #获取测试集文件名的部分作为标签
        else :#训练集和验证集标签
            label = 1 if 'dog' in img_path.split('\\')[-1] else 0
            #获取train 中文件名中的标签并进行数字化，dog为1，cat为0
        data = Image.open(img_path).convert('RGB') #open用于读取图像数据，data为Image对象
        data = self.transforms(data) #data转为tensor类型
        return data,label
train_data = MyDataset(r'D:\PyCharm\LeNet\dataset_kaggledogvscat\dataset_kaggledogvscat\train\train',train=True)
val_data = MyDataset(r'D:\PyCharm\LeNet\dataset_kaggledogvscat\dataset_kaggledogvscat\train\train',train=False)

#训练集数据
img,label = train_data[0]
print('图像的形状为{}，标签为{}'.format(img.shape,label))

train_dataloader = DataLoader(train_data,batch_size=4,shuffle=True)
for batch_datas,batch_labels in train_dataloader:
    print(batch_datas.shape,batch_labels.size())
    break
val_dataloader = DataLoader(val_data,batch_size=4,shuffle=True)
for batch_datas,batch_labels in train_dataloader:
    print(batch_datas.shape,batch_labels.size())
    break

