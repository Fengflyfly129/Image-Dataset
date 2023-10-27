'''
计算正则化的均值和标准差参数的文件
'''
import numpy as np
import cv2
import os

img_h,img_w = 32,32   #经过处理后的图片尺寸
means,stdevs = [],[]
img_list = []

imgs_path = r'D:\PyCharm\LeNet\dogs_cats\dogs_cats\data\train\dogs'
imgs_path_list = os.listdir(imgs_path)#读取文件夹中所有图片的名字，形成一个列表

len_ = len(imgs_path_list)
i = 0

for item in imgs_path_list:
    #img = cv2.imread(os.path.join(imgs_path,item))#os.path.join将字符串合并成一个字符串
    img = cv2.imread((imgs_path+'\\'+item))
    img = cv2.resize(img,(img_w,img_h))
    img = img[:,:,:,np.newaxis]#np.newaxis用于增加维度，放在哪里增加哪个维度
    img_list.append(img)
    i +=1
    print(i,'/',len_)

imgs = np.concatenate(img_list,axis=3)#将所有图片沿着新增加的维度级联在一起
imgs = imgs.astype(np.float32)/255 #归一化

for i in range(3):
    pixels = imgs[:,:,i,:].ravel()#将多维沿着颜色通道拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

#BGR->RGB，CV读取的需要转换，PIL读取的不用转换
means.reverse()#实现将整个列表或者字符串倒过来
stdevs.reverse()

print('normMean = {}'.format(means))
print('normStd = {}'.format(stdevs))




