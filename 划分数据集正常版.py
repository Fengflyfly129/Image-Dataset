import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import datasets,transforms,utils
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')
'''
不分类版数据集划分，标签在图片名字内，这样划分的效果不好
'''

def imshow(img):
    img = img/2+0.5                  #unnormalization
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))#transpose函数用于交换维度
    plt.show()
class MyDataset(Dataset):
    def __init__(self,root,transform=None,train=True,test=False):
        self.transform = transform
        self.test = test
        imag_names = [os.path.join(root,data)for data in os.listdir(root)]#保存所有图片绝对地址的列表
        if self.test == False:#训练集和验证集排序
            imag_names = sorted(imag_names,key=lambda x:int(x.split('.')[-2]))
        else :#测试集
            imag_names = sorted(imag_names,key=lambda x:int(x.split('.')[-2].split('\\')[-1]))
        imag_num = len(imag_names)
        #划分数据集
        if self.test == True:#测试集
            self.imag_names = imag_names
        elif train:#训练集
            self.imag_names = imag_names[:int(0.7*imag_num)]
        else:#验证集
            self.imag_names = imag_names[int(0.7*imag_num):]
        if self.transform is None: #None要用is 和is not 判断不能用!=
            normalize = transforms.Normalize(mean=[0.492, 0.461, 0.417],std=[0.256, 0.248, 0.251])
            if self.test == True or train == False:#测试集和验证集
               self.transform = transforms.Compose([
                   transforms.Resize(28),
                   transforms.CenterCrop(28),
                   transforms.ToTensor(),
                   normalize
               ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(28),
                    transforms.CenterCrop(28),
                    transforms.ToTensor(),
                    transforms.RandomHorizontalFlip(),
                    normalize
                ])
    def __len__(self):
        return len(self.imag_names)
    def __getitem__(self,index):
        img_path = self.imag_names[index]
        imag = Image.open(img_path)
        imag = self.transform(imag)
        if self.test:#测试集没有标签，随意取的，
            label = int(img_path.split('.')[-2].split('\\')[-1])
            #获取测试集文件名的部分作为标签
        else :#训练集和验证集标签
            label = 1 if 'dog' in img_path.split('\\')[-1] else 0
            #获取train 中文件名中的标签并进行数字化，dog为1，cat为0
        return imag,label

test_data = MyDataset(r'D:\PyCharm\LeNet\dataset_kaggledogvscat\dataset_kaggledogvscat\train\train',test=True)
dataloader = DataLoader(test_data,batch_size=4,shuffle=True)
dataiter,labels = next(iter(dataloader))

imshow(utils.make_grid(dataiter))#将多张图合并成一张图，make_grid可以将多张图片合成一张图片网格的张量，后续可通过转numpy或者转PIL进行输出显示
print(','.join(['小狗'if labels[j].item()==1 else '小猫'for j in range(4)]))#item()用于取出张量的单个值，保持类型不变

