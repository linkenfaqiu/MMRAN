import os
import random
import numpy as np
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image
from tqdm import tqdm  # 用于进度条显示

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

from sklearn.preprocessing import OneHotEncoder

from MMRAN import *
from dataset import BrainDataset
from metrics import SegmentationMetric

from medpy.metric import binary
from thop import profile

import warnings

# 以下代码为固定随机种子
def seed_torch(seed=1029):	#1029
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed) # 为了禁止hash随机化，使得实验可复现
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = A.Compose([
    # A.Normalize(mean=(0.485),std=(0.229),max_pixel_value=255.0, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    #A.GaussNoise(p=0.5),  # 将高斯噪声应用于输入图像。
    A.RandomRotate90(p=0.5),
    A.ImageCompression(quality_lower=80, quality_upper=100, p=0.5),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.3),
    # A.Emboss(alpha=(0.5, 0.7), strength=(0.2, 0.7)),
    ToTensorV2(p=1)
])

y_transforms = A.Compose([
    #A.Resize(height=256, width=256, p=1.0),
    ToTensorV2(p=1),
])

def set_requires_grad(model, layer_names, requires_grad):
    for name, param in model.named_parameters():
        if any(layer in name for layer in layer_names):
            param.requires_grad = requires_grad

def train_model(model, criterion, criterion2, optimizer, dataload, dataload_val, num_epochs=240):  
    best_val_acc = 0.0
    best_idx = -1
    no_improve_count = 0       # 记录没有提升的epoch数量
    early_stop_threshold = args.early_stop
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print('-' * 30)

        # 初始化指标
        epoch_loss = 0
        epoch_segmentaion_loss = 0
        epoch_classify_loss = 0
        correct = 0
        seg_acc = 0

        model.train()  # 切换模型为训练模式
        
        # 设置训练进度条，每 1 秒更新一次，如果要用nohup的话，可以把mininterval调大
        train_progress = tqdm(dataload, desc=f"Training Epoch {epoch + 1}", mininterval=10)

        for x, y, l in train_progress:
            inputs = x.to(device)
            labels = y.to(device)
            labels2 = (l - 1).to(device)  # 分类标签调整

            # Forward
            outputs, outputs_2 = model(inputs)
            loss1 = criterion(outputs, labels)
            loss2 = criterion2(outputs_2, labels2.long())
            loss = loss1 + loss2

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 累积损失
            epoch_loss += loss.item()
            epoch_segmentaion_loss += loss1.item()
            epoch_classify_loss += loss2.item()

            # 计算准确率
            predicted = torch.max(outputs_2, 1)[1]
            correct += (predicted == labels2).sum().item()
            seg_acc += (1 - loss1).item()

        # 计算训练集平均指标
        train_loss = epoch_loss / len(dataload)
        train_seg_loss = epoch_segmentaion_loss / len(dataload)
        train_cla_loss = epoch_classify_loss / len(dataload)
        train_cla_acc = correct / len(dataload.dataset)
        train_seg_acc = seg_acc / len(dataload)

        # 验证阶段
        epoch_val_loss = 0
        epoch_seg_val_loss = 0
        epoch_cla_val_loss = 0
        val_correct = 0
        val_seg_acc = 0

        model.eval()  # 切换模型为验证模式
        val_progress = tqdm(dataload_val, desc=f"Validation Epoch {epoch + 1}", mininterval=1)  # 验证进度条

        with torch.no_grad():
            for x, y, l in val_progress:
                inputs = x.to(device)
                labels = y.to(device)
                labels2 = (l - 1).to(device)

                outputs, outputs_2 = model(inputs)
                loss1_val = criterion(outputs, labels)
                loss2_val = criterion2(outputs_2, labels2.long())
                loss_val = loss1_val + loss2_val

                epoch_val_loss += loss_val.item()
                epoch_seg_val_loss += loss1_val.item()
                epoch_cla_val_loss += loss2_val.item()

                predicted = torch.max(outputs_2, 1)[1]
                val_correct += (predicted == labels2).sum().item()
                val_seg_acc += (1 - loss1_val).item()

        # 计算验证集平均指标
        val_loss = epoch_val_loss / len(dataload_val)
        val_seg_loss = epoch_seg_val_loss / len(dataload_val)
        val_cla_loss = epoch_cla_val_loss / len(dataload_val)
        val_cla_acc = val_correct / len(dataload_val.dataset)
        val_seg_acc = val_seg_acc / len(dataload_val)

        # 保存最佳模型 (按分类准确率判断最佳)
        ave_acc = (val_cla_acc + val_seg_acc) / 2
 
        # 保存最佳模型 (按分类准确率判断最佳)
        if ave_acc > best_val_acc:  # 如果当前平均精度更高
            best_val_acc = ave_acc
            best_idx = epoch
            torch.save(model.state_dict(), './best_model.pth')
            print(f"Best model saved at Epoch {epoch + 1} with val_acc: {best_val_acc:.6f}")
            no_improve_count = 0  # 重置计数器
        else:
            no_improve_count += 1  # 否则增加未提升计数

        # 记录训练日志
        time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        train_list = [time, epoch, train_loss, train_seg_loss, train_cla_loss, train_seg_acc, train_cla_acc]
        val_list = [time, epoch, val_loss, val_seg_loss, val_cla_loss, val_seg_acc, val_cla_acc]

        pd.DataFrame([train_list]).to_csv('./train_acc.csv', mode='a', header=False, index=False)
        pd.DataFrame([val_list]).to_csv('./val_acc.csv', mode='a', header=False, index=False)

        # 打印简洁的日志
        print(f"Train Loss: {train_loss:.6f} | Seg Loss: {train_seg_loss:.6f} | Cla Loss: {train_cla_loss:.6f} | "
              f"Seg Acc: {train_seg_acc:.6f} | Cla Acc: {train_cla_acc:.6f}")
        print(f"Val   Loss: {val_loss:.6f} | Seg Loss: {val_seg_loss:.6f} | Cla Loss: {val_cla_loss:.6f} | "
              f"Seg Acc: {val_seg_acc:.6f} | Cla Acc: {val_cla_acc:.6f}")
        
        # 检查是否达到早停条件
        if no_improve_count >= early_stop_threshold:
            print(f"Early stopping at epoch {epoch + 1}. No improvement for {early_stop_threshold} epochs.")
            break

    print(f"Training phase completed~. Best model was saved at Epoch {best_idx + 1}")
    return model

# 训练模型
def train(args):
    model = MMRAN(1, 1, args.reduce).to(device)
    
    flag = args.flag
    if flag == "Yes":
        model.load_state_dict(torch.load('best_model.pth',map_location='cuda:0'), False)     # 原先为 = 'cuda:0'

    # 输入一个 dummy 输入，假设输入尺寸为 (1, 3, 224, 224)
    input = torch.randn(1, 1, 512, 512).to(device)
    macs, params = profile(model, inputs=(input, ))

    print(f"MACs: {macs}")
    print(f"Params: {params}")

    batch_size = args.batch_size
    criterion = DiceLoss()
    criterion2 = focal_loss(alpha=0.25, gamma=2, num_classes=3, size_average=True)
    # criterion = nn.BCEWithLogitsLoss()
    # criterion2 = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-8)       # 使用默认的学习率 lr=0.001
    print(f"Current learning rate: {optimizer.param_groups[0]['lr']}")

    brain_dataset = BrainDataset("data/train", transform=x_transforms)
    dataloaders = DataLoader(brain_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    brain_val = BrainDataset("data/val", transform=y_transforms)
    dataloader_val = DataLoader(brain_val, batch_size=batch_size, shuffle=False, num_workers=2)
    train_model(model, criterion, criterion2, optimizer, dataloaders, dataloader_val)
    
#显示模型的输出结果
def test(args):
    model = MMRAN(1, 1, args.reduce).to(device)
    criterion = DiceLoss()
    model.load_state_dict(torch.load(args.ckpt,map_location='cuda:0'), False)     # 原先为 = 'cuda:0'
    brain_dataset = BrainDataset("data/test", transform=y_transforms)  # 应该要改为None
    dataloaders = DataLoader(brain_dataset, batch_size=1, shuffle=False)
    test_size = len(dataloaders)
    model.eval()

    cnt = 1
    seg_acc = 0
    cla_acc = 0
    hd95 = 0
    pred_roc = []
    label_roc = []
    mIoU_all = 0
    mpa_all = 0
    with torch.no_grad():
        for x, y, l in dataloaders:
            inputs = x.to(device)
            labels = y.to(device)
            l = l - 1            # 这里会输出batch_size个的拼接
            labels2 = l.to(device)
            #seg_res = model(inputs)
            seg_res, cla_res = model(inputs)
            loss1 = criterion(seg_res, labels)    # 计算dice指标
            seg_acc += (1 - loss1)
            predicted = torch.max(cla_res, 1)[1]
            cla_acc += (predicted == labels2).sum()

            # 第一种计算hausdorff距离的方式            
            # seg_res_a = torch.squeeze(seg_res)    # 用于计算HD
            # labels_a = torch.squeeze(labels)
            # h = hausdorff_distance(np.array(seg_res_a.cpu()), np.array(labels_a.cpu()), distance="euclidean")      # 这里没有指定像素的物理尺寸，因此在最后输出时需要×0.49
            
            # 第二种计算hausdorff距离的方式    
            h = binary.hd95(np.array(seg_res.cpu()), np.array(labels.cpu()), voxelspacing=0.49)     # voxelspacing需要指定为0.49
            hd95 += h
            
            y = seg_res.sigmoid().to(device)        # 网络最后输出时没有sigmoid，在test时才使用
            #print(y)
            #print(y.shape)
            labels = np.array(labels.cpu())
            # 下面开始画ROC曲线
            if(labels2 == 2):	# 改标签时这里要改
              label_roc += [1]
            else:
              label_roc += [0]
            
            cla_res = F.softmax(cla_res,dim=1)
            # print(cla_res)
            pred_roc += [round(cla_res[0][2].item(),3)]    # 这里也要改，[0][肿瘤类别]
            img_y = torch.squeeze(y).cpu().numpy()
            result1 = np.trunc(np.array(img_y * 255))
            result1 = result1.flatten()
            labels = labels.flatten()
            result1 = result1.astype(int)
            labels = labels.astype(int)
            for i in range(len(result1)):
                if(result1[i] > 0):
                  result1[i] = 1
            for i in range(len(labels)):
                if(labels[i] > 0):
                  labels[i] = 1
            metric = SegmentationMetric(2)  # 表示分类数量（即分割任务中的类别数）。在这里，分类数为 2（通常指背景和前景）。
            metric.addBatch(result1, labels)
            pa = metric.pixelAccuracy()
            #cpa = metric.classPixelAccuracy()
            mpa = metric.meanPixelAccuracy()
            mIoU = metric.meanIntersectionOverUnion()
            
            mIoU_all += mIoU
            mpa_all += mpa

            img_y = Image.fromarray(img_y * 255.0)      # 不乘255.0的话会输出一片黑，因为上面已经sigmoid归一化到(0,1)了
            #print(img_y)    # <PIL.Image.Image image mode=F size=512x512 at 0x7F896D90FD30>
            # 以下代码是原来的超级简陋保存图片方法
            plt.set_cmap('binary')      # 输出二值图，不指定的话背景会是紫色的
            img_y = img_y.convert('L')    #这句仍然注释，是转成二值图
            
            # 保存路径
            save_dir = './ans'
            save_path = os.path.join(save_dir, str(cnt) + '.png')

            # 检查文件夹是否存在，如果不存在则创建
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            # 保存图片
            img_y.save(save_path)

            cnt = cnt + 1
            #plt.pause(2)
            
        #plt.show()
        print(f"There are {test_size} images in the test dataset.")
        print("Test Results: segmentation accuracy is %0.6f, hd95 is %0.6f, mIoU is %0.6f, mPA is %0.6f, classification accuracy is %0.6f" % (seg_acc / test_size, hd95 / test_size, mIoU_all / test_size, mpa_all / test_size, cla_acc / test_size))

        # print(pred_roc)
        # print(label_roc)


if __name__ == '__main__':
    #参数解析
    parse=argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=10)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file", default='best_model.pth')
    parse.add_argument("--flag", type=str, help="continue to train?", default="No")
    parse.add_argument("--early_stop", type=int, help="Number of epochs to wait for improvement before stopping", default=80)
    parse.add_argument("--reduce", type=int, help="How much do you want to reduce the number of network parameters?", default=4)

    args = parse.parse_args()
    
    warnings.filterwarnings("ignore")
    seed_torch(seed=1029)   # 为保证实验可复现，设置随机数种子seed=1029，如果您需要多次训练后求测试集结果的平均值，可以更换其它seed
    
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"    # 让所有的 CUDA 操作以同步模式运行。

    if args.action=="train":
        train(args)
    elif args.action=="test":
        test(args)
