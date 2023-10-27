import torch
from torch .utils.data import DataLoader,Dataset
import torchvision
from torchvision import transforms,datasets,utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
'''
Dataset本身是torch的一个抽象类，用于创建自身的数据集，
Dataset负责建立索引到样本的映射，DataLoader负责以特定的方式从数据集中迭代的产生
一个batch的样本集合，可以使用enumerate函数，
如for batch_idx,(inputs,labels) i n enumerate(dataloader):
print(batch_idx,end=' ')
print(inputs,labels)来访问dataloader中的数据
Dataset是一个抽象类，在编写自己的数据集类时必须继承Dataset,且需重新改写__getitem__和__len__放法
__getitem__:传入指定的索引index后，该方法能够根据索引返回对应的单个样本及其对应的标签（以元组的形式）
__len__:返回整个数据集的大小
若自定义的数据集类在继承Dataset后未改写__getitem__,则程序会抛出NotImplementedError的异常
'''
data_transform = transforms.Compose([
    transforms.Resize(32),#缩放图片，最短边为32
    transforms.CenterCrop(32),#中心裁剪32*32
    transforms.ToTensor(),#转tensor
    transforms.Normalize(mean=[0.492,0.461,0.417],std=[0.256,0.248,0.251])#正则化，标准化至【-1,1】，参数为均值和标准差，参数是程序算出来的
 ])
def imshow(img):
    img = img/2+0.5                  #unnormalization
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))#transpose函数用于交换维度
    plt.show()
class MyDataset(Dataset):#继承Dataset
    def __init__(self,path_dir,transforms=None):
        self.path_dir = path_dir
        self.transform = transforms
        self.images = os.listdir(self.path_dir)#self.images内保存路径下所有文件的文件名，并存在一个列表中
    def __len__(self):#返回整个数据集的大小
        return len(self.images)
    def __getitem__(self,index):#根据索引index图像及标签
        image_index = self.images[index]
        img_path = os.path.join(self.path_dir,image_index)
        img = Image.open(img_path).convert('RGB')#读取图像，Image.convert('RGB')用于将图像转化为RGB格式,采用PIL方法读取图片
        '''
        Image.open(）得到的img数据类型呢是Image对象，不是普通的数组。
        cv2.imread()得到的img数据类型是np.array()类型
        '''

        #获取图像标签
        label = img_path.split('\\')[-1].split('.')[0]
        #提取名字中的标签
        label = 1 if 'dog' in label else 0
        if self.transform is not None:
            img = self.transform(img)#图像增强底层使用方法
        return img,label

dataset = MyDataset(r'D:\PyCharm\LeNet\dataset_kaggledogvscat\dataset_kaggledogvscat\train\train',transforms=data_transform)#此处类的实例化相当于datasets.ImageFolder函数
img,label = dataset[0]
#print(img)#PIL.Image.Image类型,可以直接输出显示
#print(label)
#plt.imshow(img)
#plt.show()
#注意，经过transform后图像会变为torch类型

#DataLoader分批加载数据
dataloader = DataLoader(dataset,batch_size=4,shuffle=True)
for batch_idx,(batch_images,batch_labels) in enumerate(dataloader):
    print(batch_images.size(),batch_labels.size())
    break

dataiter = iter(dataloader)
images,labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))#将多张图合并成一张图，make_grid可以将多张图片合成一张图片网格的张量，后续可通过转numpy或者转PIL进行输出显示
print(','.join(['小狗'if labels[j].item()==1 else '小猫'for j in range(4)]))#item()用于取出张量的单个值，保持类型不变