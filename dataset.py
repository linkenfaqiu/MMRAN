import torch
from torch.utils.data import Dataset
import PIL.Image as Image
import os
import numpy as np
import random
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
import warnings
import cv2
warnings.filterwarnings("ignore")

def make_dataset(root):
    imgs=[]
    root_1 = root + '/1/'
    root_2 = root + '/2/'
    root_3 = root + '/3/'
    root = [root_1, root_2, root_3]
    # onehot_label = [[1,0,0],[0,1,0],[0,0,1]]
    # onehot_label = torch.Tensor(onehot_label)     # 不然会报错 'list' objective has no attribute 'to'
    for i in range(3):  # 3为肿瘤的类别数
        if os.path.exists(root[i]):
            file = os.listdir(root[i])
        for filename in file:
            # 剔除名字含有 _mask 的文件
            if '_mask' in filename:
                continue

            now_name = filename
            img = root[i] + now_name

            # replace函数没有返回值，一定要赋值给一个变量
            now_name = now_name.replace('.png', '_mask.png')
            mask = root[i] + now_name

            label = i + 1
            # label = onehot_label[i]
            
            # 这里存的是文件路径
            imgs.append((img,mask,label))
            #imgs.append((img))
    return imgs

    # imgs=[]
    # n = len(os.listdir(root))//2
    # for i in range(n):
    #     img=os.path.join(root,"%d.png"% (i + 1))       # 04d
    #     mask=os.path.join(root,"%d_mask.png"% (i + 1))
    #     imgs.append((img,mask))
    #     # imgs.append((img, mask))    # 原先的
    #     #imgs.append((img))
    # return imgs

class LiverDataset(Dataset):
    def __init__(self, root, transform=None):   #, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        #self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path, label = self.imgs[index]
        # print(index)
        # print('***')
        # print(x_path)
        # print('******')
        # print(y_path)
        # print('*********')
        print("现在处理的文件为", x_path)
        #img_x = Image.open(x_path)
        #img_y = Image.open(y_path)
        # 这里才把文件从路径中取出来
        img_x = cv2.imread(x_path)
        img_y = cv2.imread(y_path)
        
        # seed = torch.manual_seed(2147483647)
        
        if self.transform is not None:
            seed = random.randint(1,1000000)
            random.seed(seed)
            # print(seed)
            aug = self.transform(image = img_x, mask = img_y)
            img_x_auged = aug['image']
            img_y_auged = aug['mask']
            img_x = img_x_auged[0,:,:]
            
            #img_x = self.transform(img_x)
            #random.seed(seed)
            # print(seed)    # 目前来看两次seed都是同一个值
        #if self.target_transform is not None:
            #img_y = self.target_transform(img_y)
            #print(img_x.shape)
            img_y = img_y_auged[:,:,0]
           
            img_x = torch.reshape(img_x, (1,img_x.shape[0],img_x.shape[1]))
            img_y = torch.reshape(img_y, (1,img_y.shape[0],img_y.shape[1]))
            img_x = img_x.float()
            img_y = (img_y / 255).float()
            #print("y:*****")
            #print(img_y.shape)
        return img_x, img_y, label

    def __len__(self):
        return len(self.imgs)