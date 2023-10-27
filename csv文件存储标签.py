import math
import os
import shutil
from collections import Counter
'''
按照类别进行测试集、验证集和训练集的划分，效果好
本质：
1、读取csv文件，标签和名字分离，建立一个对应的字典索引
2、遍历train和test文件夹，将图片通过索引复制到要求文件夹
3、在读取图片中难免遇到一些图片报错，可以使用try-except来跳过异常
'''
data_dir =r'D:\PyCharm\LeNet\dog-breed-identification'#数据集的根目录，分这么细因为要分别调取test和train中的图片
label_file = 'labels.csv'#根目录中csv文件名+后缀
train_dir = 'train'#根目录中训练集文件夹的名字
test_dir = 'test'#根目录中测试集文件夹的名字
input_dir = 'train_valid_test'#用于存放拆分后数据集的文件夹名字，可以不先创建，会自动穿件
batch_size = 4#送往训练的batch_size
valid_ratio = 0.1#将训练集划分为训练集90%验证集10%

#对文件归类
def reorg_dog_data(data_dir,label_file,train_dir,test_dir,input_dir,valid_ratio):
    #读取csv训练数据标签以及对应的文件名
    with open(os.path.join(data_dir,label_file),'r') as f:
        #跳过文件头行（栏名称）
        lines = f.readlines()[1:]#readlines会按行读取文件中所有内容，将每一行内容变为一个字符串然后，所有字符串合成一个列表
        tokens = [l.rstrip().split(',') for l in lines]#rstrip默认删除字符串末尾的空白字符.tokens为列表
        '''
        rstrip('char')会从字符串末尾开始一个个取出待替换字符串最后一个字符，与char中的字符逐个对比，有则删掉，没有则结束
        '''
        idx_label = dict(((idx,label) for idx,label in tokens))#dict输出idx为键，label为值，实现了图片名字和类别的映射
    #idx_label.values()为一个dict_value对象

    #训练集中数量最少一类的狗的数量
    min_num_train_per_label = (
        Counter(idx_label.values()).most_common()[:-2:-1][0][1])#Counter为collections的一个类，用于计算列表，元组，字符串等可迭代对象中各元素出现的次数，返回一个字典
#most_commom()为counter类的一个方法用于将counter结果转为元组，即计算各元素出现的次数，并降序排列
#本句代码意思是取most_common()转化后的元组中最小的一个狗的类别和数量，顺便将其转化为一个列表，因此采用[:-2:-1]的方式，然后取该列表中元组中的种类出现的数量
#print(Counter(idx_label.values()).most_common()[:-2:-1][0][1])这两句代码等价
#print(Counter(idx_label.values()).most_common()[-1][1])
#验证集中每类狗的数量
    num_valid_per_label = math.floor(min_num_train_per_label*valid_ratio)
    label_count = dict()#label_count用于存储验证集中各类图片个数

    def mkdir_if_not_exist(path):#判断是否有存放拆分后数据集的文件夹，没有就创建一个
        if not os.path.exists(os.path.join(*path)):#若文件夹存在则返回True
            os.makedirs(os.path.join(*path))#创建文件
    #整理训练和验证集，将数据集进行拆分复制到预先设置好的存放文件夹中。
    for train_file in os.listdir(os.path.join(data_dir,train_dir)):#train+valid
        idx = train_file.split('.')[0]
        label = idx_label[idx]
        mkdir_if_not_exist([data_dir,input_dir,'train_valid',label])
        try:
            shutil.copy(os.path.join(data_dir,train_dir,train_file),os.path.join(data_dir,input_dir,'train_valid',label))
        #shutil.copy为python自带函数用于将源文件的内容复制到目标文件中去
        except:
            pass

        if label not in label_count or label_count[label]<num_valid_per_label:#label_count字典用于统计各类别狗的数量，顺便创建验证集文件
            mkdir_if_not_exist([data_dir,input_dir,'valid',label])
            try:#try,except可以用来忽略异常报错
                shutil.copy(os.path.join(data_dir,train_dir,train_file),os.path.join(data_dir,input_dir,'valid',label))
                label_count[label]=label_count.get(label,0)+1#label键，dict.get(a,b),a为键值，如果a存在，则返回dict[a],否则返回b，b则返回None
            except:
                pass
        else:#先把数据分给验证集，验证集满了之后再给训练集
            mkdir_if_not_exist([data_dir, input_dir, 'train', label])
            try:
                shutil.copy(os.path.join(data_dir, train_dir, train_file),
                        os.path.join(data_dir, input_dir, 'train', label))
            except:
                pass
    #整理测试集，将测试集复制存放在新建路径下的unknown文件夹中
    mkdir_if_not_exist([data_dir, input_dir, 'test', 'unknown'])
    for test_file in os.listdir(os.path.join(data_dir,test_dir)):
        try:
            shutil.copy(os.path.join(data_dir, train_dir, test_file),
                    os.path.join(data_dir, input_dir, 'test','unknown'))
        except:
            pass

    print('valid_num_affenpinscher:',len(os.listdir(os.path.join(r'D:\PyCharm\LeNet\dog-breed-identification\train_valid_test\valid', 'affenpinscher'))))
#载入数据，进行数据拆分
reorg_dog_data(data_dir,label_file,train_dir,test_dir,input_dir,valid_ratio)

#数据加载
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms,datasets,utils
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
def image_show(img):
    img = img/2+0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(img,(1,2,0)))
    plt.show()
    return img

data_transform = transforms.Compose([transforms.Resize(32),
                                    transforms.CenterCrop(32),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.492,0.461,0.417],std=[0.256,0.248,0.251])
                                     ])

new_data_dir = r'D:\PyCharm\LeNet\dog-breed-identification\train_valid_test'
train_ds = datasets.ImageFolder(root = os.path.join(new_data_dir,'train'),transform=data_transform)
valid_ds = datasets.ImageFolder(root = os.path.join(new_data_dir,'valid'),transform=data_transform)

img,label = train_ds[0]
print(img)
print(img.shape)
print(label)









