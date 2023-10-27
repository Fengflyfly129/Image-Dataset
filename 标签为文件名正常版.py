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
dataset = datasets.ImageFolder(root=r'D:\PyCharm\LeNet\dogs_cats\dogs_cats\data\train',transform=data_transform)
dataloader = torch.utils.data.DataLoader(dataset,batch_size=8,shuffle=True)
#显示图片
imags,labels= next(iter(dataloader))#next(iter)取出的是一个批次里面的所有图片的一个tensor
image_show(torchvision.utils.make_grid(imags))
print(','.join(['小狗'if labels[j].item()==1 else '小猫'for j in range(8)]))


