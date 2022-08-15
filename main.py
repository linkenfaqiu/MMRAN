import pandas as pd
import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from unet import *
from dataset import LiverDataset
import random
import imageio  # 转换成图像
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import heatmap
warnings.filterwarnings("ignore")
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from thop import profile
import torch.nn.functional as F
from hausdorff import hausdorff_distance
from medpy.metric import binary
from metrics import SegmentationMetric
#import surface_distance as surfdist
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

seed_torch()


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

x_transforms = A.Compose([
    # A.Normalize(mean=(0.485),std=(0.229),max_pixel_value=255.0, p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    #A.GaussNoise(p=0.5),  # 将高斯噪声应用于输入图像。
    A.RandomRotate90(p=0.5),
    A.ImageCompression(quality_lower=80, quality_upper=100, p=0.5),
    A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=0.3),
    A.Emboss(alpha=(0.5, 0.7), strength=(0.2, 0.7)),
    ToTensorV2(p=1)
])
#x_transforms = transforms.Compose([    # transforms的方法
    # 加几种方法
    # transforms.ToTensor(),
    # transforms.Normalize([0.5], [0.5]),     # 分别为对应通道的 mean 和 std

    # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
    # 是只有一个通道的灰度图，所以需要改成下面这样。
    # transforms.Normalize((0.5,), (0.5,))  #对数据按通道进行标准化，即先减均值，再除以标准差,分别为 h w c
    # transforms.Resize(512),
    # 针对RGB空间
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    # transforms.ToPILImage(),
    # transforms.RandomResizedCrop(size=224, scale=(0.5, 0.5)),
    # transforms.RandomRotation(15, center=(0, 0), expand=True),   # expand only for center rotation
    # transforms.RandomRotation(30, center=(0, 0), expand=True),   # expand only for center rotation
    # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    # transforms.ToTensor(),
    # transforms.Resize(256),
    #transforms.ToTensor(),
    #transforms.Normalize([0.5], [0.5]),
    #transforms.ToPILImage(),
    #transforms.RandomRotation(90, resample=False, expand=False, center=None),
    #transforms.RandomHorizontalFlip(p1),
    #transforms.RandomVerticalFlip(p2),
    #transforms.ToTensor(),
#])
y_transforms = A.Compose([
    #A.Resize(height=256, width=256, p=1.0),
    ToTensorV2(p=1),
])


# mask只需要转换为tensor
# y_transforms = transforms.Compose([
# transforms.Resize([512,512]),
# transforms.ToTensor()
# ])

def train_model(model, criterion, criterion2, optimizer, dataload, dataload_val, num_epochs=240):  # 原先epoch=20
    best_loss = 10000.0
    best_idx = -1
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        epoch_segmentaion_loss = 0
        epoch_classify_loss = 0

        classify_list = []
        segmentation_list = []
        total_list = []
        step = 0
        # for循环内处理一个epoch
        correct = 0
        model.train()  # 切换模型为训练模式
        batch_idx = 0
        seg_acc = 0
        '''
        if epoch <= 70:
            for name, param in model.named_parameters():
                if "conv11" in name:  # 冻结分类分支
                    param.requires_grad = False
                if "conv12" in name:
                    param.requires_grad = False
                if "conv13" in name:
                    param.requires_grad = False
                if "conv14" in name:
                    param.requires_grad = False
                if "fc1" in name:
                    param.requires_grad = False
                if "fc2" in name:
                    param.requires_grad = False

        if epoch > 70 and epoch <= 140:
            for name, param in model.named_parameters():
                if "up6" in name:  # 冻结分割分支
                    param.requires_grad = False
                if "conv6" in name:
                    param.requires_grad = False
                if "up7" in name:
                    param.requires_grad = False
                if "conv7" in name:
                    param.requires_grad = False
                if "up8" in name:
                    param.requires_grad = False
                if "conv8" in name:
                    param.requires_grad = False
                if "up9" in name:
                    param.requires_grad = False
                if "conv9" in name:
                    param.requires_grad = False
                if "conv10" in name:
                    param.requires_grad = False
               
                if "conv11" in name:  # 解冻分类分支
                    param.requires_grad = True
                if "conv12" in name:
                    param.requires_grad = True
                if "conv13" in name:
                    param.requires_grad = True
                if "conv14" in name:
                    param.requires_grad = True
                if "fc1" in name:
                    param.requires_grad = True
                if "fc2" in name:
                    param.requires_grad = True

        if epoch > 140:
            for name, param in model.named_parameters():
                if "up6" in name:  # 解冻分割分支
                    param.requires_grad = True
                if "conv6" in name:
                    param.requires_grad = True
                if "up7" in name:
                    param.requires_grad = True
                if "conv7" in name:
                    param.requires_grad = True
                if "up8" in name:
                    param.requires_grad = True
                if "conv8" in name:
                    param.requires_grad = True
                if "up9" in name:
                    param.requires_grad = True
                if "conv9" in name:
                    param.requires_grad = True
                if "conv10" in name:
                    param.requires_grad = True

                if "conv11" in name:  # 解冻分类分支
                    param.requires_grad = True
                if "conv12" in name:
                    param.requires_grad = True
                if "conv13" in name:
                    param.requires_grad = True
                if "conv14" in name:
                    param.requires_grad = True
                if "fc1" in name:
                    param.requires_grad = True
                if "fc2" in name:
                    param.requires_grad = True
        '''
        for x, y, l in dataload:
            batch_idx += 1
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            l = l - 1
            labels2 = l.to(device)  # 这个是分类标签
            # zero the parameter gradients
            # forward
            outputs, outputs_2 = model(inputs)  # 分割网络的结果
            # labels = torch.LongTensor(labels)
            loss1 = criterion(outputs, labels)  # seg_loss
            loss2 = criterion2(outputs_2, labels2.long())  # cla_loss
            # loss2 = 1 - outputs_2[int(labels)]
            optimizer.zero_grad()
            # loss = (1 - epoch / num_epochs) * loss1 + epoch / num_epochs * loss2    #交替训练
            loss = loss1 + loss2    #直接训练
            loss.backward()
            # loss2.backward()
            optimizer.step()
            # epoch_loss += loss.item()
            epoch_loss += loss.item()
            epoch_segmentaion_loss += loss1.item()
            epoch_classify_loss += loss2.item()
            predicted = torch.max(outputs_2, 1)[1]
            correct += (predicted == labels2).sum()
            seg_acc += (1 - loss1)
            print("%d/%d,train_loss:%0.8f,seg_loss:%0.8f,cla_loss:%0.8f" % (step,
                  (dt_size - 1) // dataload.batch_size + 1,loss.item(), loss1.item(), loss2.item()))

        epoch_val_loss = 0
        epoch_seg_val_loss = 0
        epoch_cla_val_loss = 0
        val_size = len(dataload_val.dataset)
        val_seg_acc = 0
        val_correct = 0
        val_step = 0
        model.eval()  # 切换模型为验证模式
        with torch.no_grad():  # 不记录模型梯度信息
            for x, y, l in dataload_val:
                val_step += 1
                inputs = x.to(device)
                labels = y.to(device)
                l = l - 1  # 这里会输出batch_size个的拼接
                labels2 = l.to(device)
                outputs, outputs_2 = model(inputs)  # 分割网络的结果
                loss1_val = criterion(outputs, labels)  # seg_loss
                loss2_val = criterion2(outputs_2, labels2.long())  # cla_loss
                loss_val = loss1_val + loss2_val
                epoch_val_loss += loss_val.item()
                epoch_seg_val_loss += loss1_val.item()
                epoch_cla_val_loss += loss2_val.item()
                predicted = torch.max(outputs_2, 1)[1]
                val_correct += (predicted == labels2).sum()
                val_seg_acc += (1 - loss1_val)

                print("%d/%d,val_loss:%0.8f,val_seg_loss:%0.8f,val_cla_loss:%0.8f" % (val_step,
                      (val_size - 1) // dataload_val.batch_size + 1,loss_val.item(), loss1_val.item(),loss2_val.item()))

        if (epoch_val_loss / val_step) < best_loss:
            best_loss = epoch_val_loss / val_step
            best_idx = epoch
            torch.save(model.state_dict(), './best_model.pth')
            print("当前最佳模型已保存，是第%d个epoch时的模型" % best_idx)

        print("当前epoch训练精度为")
        print("epoch %d loss:%0.8f, " % (epoch, epoch_loss / step),
              "segementation_loss:%0.8f, " % (epoch_segmentaion_loss / step),
              "classify_loss:%0.8f" % (epoch_classify_loss / step),
              "seg_acc:%0.8f" % (seg_acc * dataload.batch_size / (dt_size - 1)),
              "cla_acc:%0.8f" % (correct / (dt_size - 1)))
        
        time = "%s" % datetime.now()  # 获取当前时间
        Step = "Step[%d]" % epoch
        train_list = [time, Step, str(epoch_loss / step), str(epoch_segmentaion_loss / step),
                      str(epoch_classify_loss / step), \
                      str(seg_acc * dataload.batch_size / (dt_size - 1)), str(correct / (dt_size - 1))]
        
        data = pd.DataFrame([train_list])
        data.to_csv('./train_acc.csv', mode='a', header=False, index=False)
        
        print("当前epoch验证精度为")
        print("epoch %d loss:%0.8f, " % (epoch, epoch_val_loss / val_step),
              "segementation_loss:%0.8f, " % (epoch_seg_val_loss / val_step),
              "classify_loss:%0.8f" % (epoch_cla_val_loss / val_step),
              "val_seg_acc:%0.8f" % (val_seg_acc * dataload_val.batch_size / (val_size - 1)),
              "val_cla_acc:%0.8f" % (val_correct / (val_size - 1)))
        
        val_list = [time, Step, str(epoch_val_loss / step), str(epoch_seg_val_loss / step),
                    str(epoch_cla_val_loss / step), \
                    str(val_seg_acc * dataload_val.batch_size / (val_size - 1)), str(val_correct / (val_size - 1))]

        data_val = pd.DataFrame([val_list])
        data_val.to_csv('./val_acc.csv', mode='a', header=False, index=False)

        classify_list.append(epoch_classify_loss)
        segmentation_list.append(epoch_segmentaion_loss)
        total_list.append(epoch_loss)
        
    torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
    print("最佳模型已保存，是第%d个epoch的模型" % best_idx)

    return model

# 训练模型
def train(args):
    model = Unet(1, 1).to(device)
    flag = args.flag
    if flag:
      model.load_state_dict(torch.load('best_model.pth',map_location='cuda:0'), False)     # 原先为 = 'cuda:0'

    # 以下三行为计算参数量代码
    #inputs = torch.randn(1, 1, 512, 512).to(device)
    #flops, params = profile(model, (inputs,))
    #print('flops: ', flops, 'params: ', params)

    batch_size = args.batch_size
    criterion = DiceLoss()
    criterion2 = focal_loss(alpha=0.25, gamma=2, num_classes=3, size_average=True)
    # criterion = nn.BCEWithLogitsLoss()
    # criterion2 = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-8)
    liver_dataset = LiverDataset("data/train", transform=x_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    brain_val = LiverDataset("data/val", transform=y_transforms)
    dataloader_val = DataLoader(brain_val, batch_size=batch_size, shuffle=False, num_workers=2)
    train_model(model, criterion, criterion2, optimizer, dataloaders, dataloader_val)
    
#显示模型的输出结果
def test(args):
    model = Unet(1, 1).to(device)
    criterion = DiceLoss()
    model.load_state_dict(torch.load(args.ckpt,map_location='cuda:0'), False)     # 原先为 = 'cuda:0'
    liver_dataset = LiverDataset("data/test", transform=y_transforms)  # 应该要改为None
    dataloaders = DataLoader(liver_dataset, batch_size=1, shuffle=False)
    test_size = len(dataloaders)
    model.eval()
    #test = GradCam(model, 'conv5')
    #print(test.numpy())
    #plt.ion()
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
            seg_res_a = torch.squeeze(seg_res)    # 用于计算HD
            labels_a = torch.squeeze(labels)
            #h = hausdorff_distance(np.array(seg_res_a.cpu()), np.array(labels_a.cpu()), distance="euclidean")
            h = binary.hd95(np.array(seg_res.cpu()), np.array(labels.cpu()), voxelspacing=0.45)
            print(h)
            hd95 += h
            #surface_distances = surfdist.compute_surface_distances(seg_res.cpu(), labels.cpu(), spacing_mm=(0.45, 0.45))
            #print(h)
            #binary.hd95(x, labels, voxelspacing=None)
            y = seg_res.sigmoid().to(device)        # Unet网络最后输出时没有sigmoid，在test时才使用
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
            result1 = result1.astype(np.int)
            labels = labels.astype(np.int)
            for i in range(len(result1)):
                if(result1[i] > 0):
                  result1[i] = 1
            for i in range(len(labels)):
                if(labels[i] > 0):
                  labels[i] = 1
            metric = SegmentationMetric(2)  # 3表示有3个分类，有几个分类就填几
            metric.addBatch(result1, labels)
            pa = metric.pixelAccuracy()
            #cpa = metric.classPixelAccuracy()
            mpa = metric.meanPixelAccuracy()
            mIoU = metric.meanIntersectionOverUnion()
            #print('pa is : %f' % pa)
            #print('cpa is :')  # 列表
            #print(cpa)
            #print('mpa is : %f' % mpa)
            #print('mIoU is : %f' % mIoU)
            mIoU_all += mIoU
            mpa_all += mpa
            #result1 = (result1-np.min(result1))/(np.max(result1)-np.min(result1))
            #np.savetxt('npresult1.txt',result1)
            #img = r'./data/test/1/268.png'
            
            #heatmap.visulize_attention_ratio(img, result1)
            #print(img_y.shape)
            #rint("第" + str(cnt) + "张图的结果为：")
            #print("分类概率为：")
            #print(cla_res)
            #print("分类结果为：")
            #print(np.argmax(cla_res) + 1)
            #print("1 表示脑膜瘤，2 表示神经胶质瘤，3 表示垂体瘤")
            #print("----" * 10)
            # imageio.imwrite('D:/study/医学图像处理/unet-liver/u_net_liver/' + str(cnt) + '.png', img_y)
            img_y = Image.fromarray(img_y * 255.0)      # 不乘255.0的话会输出一片黑，因为上面已经sigmoid归一化到(0,1)了
            #print(img_y)    # <PIL.Image.Image image mode=F size=512x512 at 0x7F896D90FD30>
            # 以下代码是原来的超级简陋保存图片方法
            plt.set_cmap('binary')      # 输出二值图，不指定的话背景会是紫色的
            img_y = img_y.convert('L')    #这句仍然注释，是转成二值图
            img_y.save('./ans/test' + str(cnt) + '.png')
            #plt.figure(figsize=(64, 64))
            #plt.axis('off')     # 取消matplotlib输出图片的坐标轴
            #plt.imshow(img_y)
            #plt.savefig(fname="test" + str(cnt) + ".png", dpi = 8)
            cnt = cnt + 1
            #plt.pause(2)
            
        #plt.show()
        print("测试时分割精度为%0.8f，hd95为%0.8f, mIoU为%0.8f, mpa为%0.8f, 分类精度为%0.8f" % (seg_acc / test_size, hd95 / test_size / 100 * 0.49, mIoU_all/test_size, mpa_all/test_size, cla_acc / test_size))
        print(pred_roc)
        print(label_roc)
        #print("测试时分割精度为%0.8f" % (seg_acc / test_size))


if __name__ == '__main__':
    #参数解析
    parse=argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    parse.add_argument("action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=10)     #这里设置了默认的batch_size,原先是10
    parse.add_argument("--ckpt", type=str, help="the path of model weight file",default='best_model.pth')
    parse.add_argument("--flag", type=bool, help="continue to train?",default=True)
    args = parse.parse_args()

    if args.action=="train":
        train(args)
    elif args.action=="test":
        test(args)
