import torch
from torch.utils.data import DataLoader,Dataset
import torchvision
from torchvision import datasets,utils,transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import PIL.Image
import math
import shutil
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

data_dir = r'D:\PyCharm\LeNet\dog-breed-identification'
label_file = 'labels.csv'
train_dir = 'train'
test_dir = 'test'
input_dir = 'train_valid_test'
batch_size =4
valid_ratio = 0.1

def reorg_dog_data(data_dir,label_file,train_dir,test_dir,input_dir,valid_ratio):
    with open(os.path.join(data_dir,label_file),'r') as f:
        lines = f.readlines()[1:]
        tokens = [l.rstrip().split(',') for l in lines]
        idx_label = dict(((idx, label) for idx, label in tokens))
        min_num_train_per_label = Counter(idx_label.values()).most_common()[-1][1]
        num_valid_per_label = math.floor(min_num_train_per_label*valid_ratio)
        label_count = dict()

    def mkdir_if_not_exist(path):
        if not os.path.exists(os.path.join(*path)):
            os.makedirs(os.path.join(*path))

    for train_file in os.listdir(os.path.join(data_dir,train_dir)):
        idx = train_file.split('.')[0]
        label = idx_label[idx]
        mkdir_if_not_exist([data_dir, input_dir, 'train_valid', label])
        try:
            shutil.copy(os.path.join(data_dir, train_dir, train_file),
                    os.path.join(data_dir, input_dir, 'train_valid', label))
        # shutil.copy为python自带函数用于将源文件的内容复制到目标文件中去
        except:
            pass
        if label not in label_count or label_count[label] < num_valid_per_label:
            mkdir_if_not_exist([data_dir, input_dir, 'valid', label])
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
        mkdir_if_not_exist([data_dir, input_dir, 'test', 'unknown'])
        for test_file in os.listdir(os.path.join(data_dir, test_dir)):
            try:
                shutil.copy(os.path.join(data_dir, train_dir, test_file),
                            os.path.join(data_dir, input_dir, 'test', 'unknown'))
            except:
                pass

reorg_dog_data(data_dir,label_file,train_dir,test_dir,input_dir,valid_ratio)

new_data_dir = r'D:\PyCharm\LeNet\dog-breed-identification\train_valid_test'
train_ds = datasets.ImageFolder(root = os.path.join(new_data_dir,'train'),transform=data_transform)
valid_ds = datasets.ImageFolder(root = os.path.join(new_data_dir,'valid'),transform=data_transform)



