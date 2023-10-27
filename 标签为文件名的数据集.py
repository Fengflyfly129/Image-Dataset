from torch.utils.data import DataLoader,Dataset #用于对数据集进行分批加载
import matplotlib.pyplot as plt
import torch
from torchvision import transforms,utils,datasets #用于对公开数据集进行读取并预处理
from PIL import Image
import pandas as pd
import numpy as np
import warnings
import torchvision
warnings.filterwarnings('ignore')#关闭warning,好看
#%matplotlib inline,这句话用于jupyter显示图像，pycharm用会报错


#通过原始文件批量显示图片
def show_images(num_rows,num_cols,titles=None):
    axes = []
    num = num_rows*num_cols
    fig = plt.figure()
    for i in range(num):
        i = fig.add_subplot(num_rows,num_cols,i+1)#figsize为生成的axes大小
        axes.append(i)
    for i in range(num):
        for ax,(img,label) in zip(axes,hymenoptera_dataset):#图片目前直接用for遍历，只能遍历文件夹dataset
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])#关闭刻度显示
            ax.spines['right'].set_color('none')#关闭图片边框
            ax.spines['top'].set_color('none')
            ax.spines['left'].set_color('none')
            ax.spines['bottom'].set_color('none')
            ax.text(15,50,label,color = 'red')
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()#显示图片，必须加这一句
    return axes

#dataloader后批量显示图片,图片三通道为(颜色，宽度，高度)
def imshow(img):
    img = img/2+0.5                  #unnormalization
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg,(1,2,0)))#transpose函数用于交换维度
    plt.show()



data_transform = transforms.Compose([
    transforms.Resize(32),#缩放图片，最短边为32
    transforms.CenterCrop(32),#中心裁剪32*32
    transforms.ToTensor(),#转tensor
    transforms.Normalize(mean=[0.492,0.461,0.417],std=[0.256,0.248,0.251])#正则化，标准化至【-1,1】，参数为均值和标准差，参数是程序算出来的
 ])
#imagefolder读取数据默认按照文件夹顺序自上往下为种类命名数字标签
hymenoptera_dataset = datasets.ImageFolder(root=r'D:\PyCharm\LeNet\dogs_cats\dogs_cats\data\train',transform=data_transform)#读数数据集并预处理
#show_images(3,3)
'''
for img,label in hymenoptera_dataset:
    print('图像img的形状{},标签label的值{}'.format(img.shape,label))
    print('图像预处理后:\n',img)
    break
'''
dataset_loader = torch.utils.data.DataLoader(hymenoptera_dataset,batch_size=4,shuffle=True)#数据划分batch_size,输出为dataloader对象
#需要使用iter()函数转为迭代器然后用next()读取

dataiter = iter(dataset_loader)
images,labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))#将多张图合并成一张图，make_grid可以将多张图片合成一张图片网格的张量，后续可通过转numpy或者转PIL进行输出显示
print(','.join(['小狗'if labels[j].item()==1 else '小猫'for j in range(4)]))#item()用于取出张量的单个值，保持类型不变
