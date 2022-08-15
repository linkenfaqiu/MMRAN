import torch
from torch import nn
import torch.nn.functional as F
from SPP_Layer import SPPLayer
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
# 以下为带有1×1卷积的代码
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1, padding=0),
            nn.GroupNorm(64, in_ch),
            #nn.BatchNorm2d(in_ch // 2),  # 已添加BN层
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(64, out_ch),
            #nn.BatchNorm2d(out_ch),  # 已添加BN层
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch // 2, 1, padding=0),
            #nn.BatchNorm2d(out_ch // 2),  # 已添加BN层
            nn.GroupNorm(64, out_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch // 2, out_ch, 3, padding=1),
            nn.GroupNorm(64, out_ch),
            #nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            #nn.BatchNorm2d(out_ch)
            nn.GroupNorm(64, out_ch),
        )

    def forward(self, input):
        out = self.conv(input)
        out = out + self.shortcut(input)
        out = F.relu(out)
        return out

'''
# 以下为无1×1卷积代码
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            #nn.BatchNorm2d(out_ch),  # 已添加BN层
            nn.GroupNorm(64, out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            #nn.BatchNorm2d(out_ch),
            nn.GroupNorm(64, out_ch),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            #nn.BatchNorm2d(out_ch)
            nn.GroupNorm(64, out_ch),
        )

    def forward(self, input):
        out = self.conv(input)
        out = out + self.shortcut(input)
        out = F.relu(out)
        return out

'''
# basic
class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.Conv_Squeeze = nn.Conv2d(in_channels,
                                      in_channels // 2,
                                      kernel_size=1)
        self.Conv_Excitation = nn.Conv2d(in_channels // 2,
                                         in_channels,
                                         kernel_size=1)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        z = self.avgpool(U)  # shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z)  # shape: [bs, c/2, 1, 1]
        z = self.Conv_Excitation(z)  # shape: [bs, c, 1, 1]
        z = self.norm(z)
        return U * z.expand_as(U)
'''
      # double
class sSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.Conv1x1 = nn.Conv2d(in_channels, 1, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()

    def forward(self, U):
        q = self.Conv1x1(U)  # U:[bs,c,h,w] to q:[bs,1,h,w]
        q = self.norm(q)
        return U * q  # 广播机制

class cSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.# new
        self.Conv_Squeeze = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.Conv_Excitation = nn.Conv2d(in_channels//2, in_channels, kernel_size=1, bias=False)
        self.norm = nn.Sigmoid()
        self.maxpool = nn.AdaptiveMaxPool2d(1, return_indices=False)

    def forward(self, U):
        z = self.avgpool(U)# shape: [bs, c, h, w] to [bs, c, 1, 1]
        z = self.Conv_Squeeze(z) # shape: [bs, c/2]
        z = self.Conv_Excitation(z) # shape: [bs, c]
        z = self.norm(z)
        x = self.maxpool(U)
        x = self.Conv_Squeeze(x)
        x = self.Conv_Excitation(x)
        x = self.norm(x)
        x = z + x
        return U * x.expand_as(U)

class scSE(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U_cse = self.cSE(U)
        U_sse = self.sSE(U_cse)
        #print((U_cse+U_sse).shape)
        return U_sse + U
      
# 定义SE模块
class SE(nn.Module):
    def __init__(self, nin, nout, reduce=16):
        super(SE, self).__init__()
        self.gp = nn.AvgPool2d(1)
        self.se = nn.Sequential(nn.Linear(nout, nout // reduce),
                                nn.ReLU(inplace=True),
                                nn.Linear(nout // reduce, nout),
                                nn.Sigmoid())
    def forward(self, input):
        x = input
        b, c, _, _ = x.size()
        y = self.gp(x).view(b, c)
        y = self.se(y).view(b, c, 1, 1)
        y = x * y.expand_as(x)
        out = y + input
        return out
      
class Unet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Unet, self).__init__()
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 512)
        self.psp = PSPModule(512, 1024, (1, 2, 3, 6))
        self.psp2 = PSPModule2(128, 128, (1, 2, 3, 6))
        # 上采样使用blinear或者转置卷积扩大特征图宽高为两倍
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        # 1 * 1 卷积，通道由64转为 out_ch（类别数，其实应该是2？），即每一个类别对应一个mask
        self.conv10 = nn.Conv2d(64, out_ch, 1)
        self.num_levels = 4
        self.pool_type = 'max_pool'
        # CNN通道接在c5后面
        self.conv11 = DoubleConv(1024, 512)  # 32 * 32 * 512
        self.pool5 = nn.MaxPool2d(2)  # 16 * 16 * 512, 注意这个函数括号中的值！！！
        self.conv12 = DoubleConv(512, 128)  # 16 * 16 * 256
        self.pool6 = nn.MaxPool2d(2)  # 8 * 8 * 256
        self.conv13 = DoubleConv(256, 128)  # 8 * 8 * 128
        self.pool7 = nn.MaxPool2d(2)  # 4 * 4 * 128
        self.conv14 = DoubleConv(128, 64)  # 4 * 4 * 64
        self.fc1 = nn.Linear(1920, 100)
        # self.fc1 = nn.Linear(1344, 100)
        self.fc2 = nn.Linear(100, 3)

    def forward(self, x):
        c1 = self.conv1(x)  # 512 * 512 * 64
        #下面两行有些模型没有
        c_se = scSE(64).to(device)
        c1 = c_se(c1).to(device)
        p1 = self.pool1(c1)  # 256 * 256 * 64
        #p1 = self.psp(c1)
        c2 = self.conv2(p1)  # 256 * 256 * 128
        c_se = scSE(128).to(device)  # 要加上.cuda()，否则数据类型不一致
        #c2 = c_se(c2).cuda()
        #c_se = SE(128, 128).cuda()
        c2 = c_se(c2).to(device)
        p2 = self.pool2(c2)  # 128 * 128 * 128
        c3 = self.conv3(p2)  # 128 * 128 * 256
        c_se2 = scSE(256).to(device)  # 要加上.cuda()，否则数据类型不一致
        #c_se2 = SE(256, 256).cuda()
        c3 = c_se2(c3).to(device)
        p3 = self.pool3(c3)  # 64 * 64 * 256
        c4 = self.conv4(p3)  # 64 * 64 * 512
        c_se3 = scSE(512).to(device)  # 要加上.cuda()，否则数据类型不一致
        #c_se3 = SE(512, 512).cuda()
        c4 = c_se3(c4).to(device)
        #print(c4.shape)
        p4 = self.pool4(c4)  # 32 * 32 * 512
        c5 = self.conv5(p4)  # 32 * 32 * 1024
        c5 = self.psp(c5)      #在网络最底层增加了多尺度融合
        # 以下开始上采样
        up_6 = self.up6(c5)  # 64 * 64 * 512
        merge6 = torch.cat([up_6, c4], dim=1)  # 64 * 64 * 1024
        c6 = self.conv6(merge6)  # 64 * 64 * 512
        up_7 = self.up7(c6)  # 128 * 128 * 256
        merge7 = torch.cat([up_7, c3], dim=1)  # 128 * 128 * 512
        c7 = self.conv7(merge7)  # 128 * 128 * 256
        up_8 = self.up8(c7)  # 256 * 256 * 128
        merge8 = torch.cat([up_8, c2], dim=1)  # 256 * 256 * 256
        c8 = self.conv8(merge8)  # 256 * 256 * 128
        up_9 = self.up9(c8)  # 512 * 512 * 64
        merge9 = torch.cat([up_9, c1], dim=1)  # 512 * 512 * 128
        c9 = self.conv9(merge9)  # 512 * 512 * 64
        c10 = self.conv10(c9)  # 512 * 512 * out_ch
        # CNN部分
        c11 = self.conv11(c5)  # 32 * 32 * 512
        p5 = self.pool5(c11)  # 16 * 16 * 512
        c12 = self.conv12(p5)  # 16 * 16 * 256
        c12n = self.psp2(c12)
        c12 = torch.cat([c12, c12n], dim = 1)
        p6 = self.pool6(c12)  # 8 * 8 * 256
        c13 = self.conv13(p6)  # 8 * 8 * 128
        p7 = self.pool7(c13)  # 4 * 4 * 128
        c14 = self.conv14(p7)  # 4 * 4 * 64
        # print(c14.shape)
        spp_layer = SPPLayer(self.num_levels, self.pool_type)
        spp = spp_layer(c14)
        # print(spp.size())
        # c14 = c14.view(-1, 64 * 4 * 4)
        f1 = F.relu(self.fc1(spp))
        # f2 = nn.Sigmoid()(self.fc2(f1))
        f2 = self.fc2(f1)
        # out = nn.Sigmoid()(c10)
        # c10 = nn.Sigmoid()(c10)
        return c10, f2

class focal_loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=3, size_average=True):
        """
        focal_loss损失函数, -α(1-yi)**γ *ce_loss(xi,yi)
        步骤详细的实现了 focal_loss损失函数.
        :param alpha:   阿尔法α,类别权重.      当α是列表时,为各类别权重,当α为常数时,类别权重为[α, 1-α, 1-α, ....],常用于 目标检测算法中抑制背景类 , retainnet中设置为0.25
        :param gamma:   伽马γ,难易样本调节参数. retainnet中设置为2
        :param num_classes:     类别数量
        :param size_average:    损失计算方式,默认取均值
        """

        super(focal_loss, self).__init__()
        self.size_average = size_average
        if isinstance(alpha, list):
            assert len(alpha) == num_classes  # α可以以list方式输入,size:[num_classes] 用于对不同类别精细地赋予权重
            print("Focal_loss alpha = {}, 将对每一类权重进行精细化赋值".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} ,将对背景类进行衰减,请在目标检测任务中使用 --- ".format(alpha))
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1 - alpha)  # α 最终为 [ α, 1-α, 1-α, 1-α, 1-α, ...] size:[num_classes]
        self.gamma = gamma

    def forward(self, preds, labels):
        """
        focal_loss损失计算
        :param preds:   预测类别. size:[B,N,C] or [B,C]    分别对应与检测与分类任务, B 批次, N检测框数, C类别数
        :param labels:  实际类别. size:[B,N] or [B]
        :return:
        """
        # assert preds.dim()==2 and labels.dim()==1
        preds = preds.view(-1, preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        preds_softmax = F.softmax(preds,
                                  dim=1)  # 这里并没有直接使用log_softmax, 因为后面会用到softmax的结果(当然你也可以使用log_softmax,然后进行exp操作)
        preds_logsoft = torch.log(preds_softmax)
        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))  # 这部分实现nll_loss ( crossempty = log_softmax + nll )
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        self.alpha = self.alpha.gather(0, labels.view(-1))
        loss = -torch.mul(torch.pow((1 - preds_softmax), self.gamma),
                          preds_logsoft)  # torch.pow((1-preds_softmax), self.gamma) 为focal loss中 (1-pt)**γ
        loss = torch.mul(self.alpha, loss.t())
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score
      
 # PSP模块
class PSPModule(nn.Module):
    def __init__(self, features, out_features, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        # prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        prior = nn.AdaptiveMaxPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)    #第一次加入多尺度模块时没加1*1卷积层，但是精度也有不错的提升
        return nn.Sequential(prior, conv)
        #return nn.Sequential(prior)
    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))    # 1代表cat按列拼
        return self.relu(bottle)

class PSPModule2(nn.Module):
    def __init__(self, features, out_features, size=(1,2,3,6)):
        super().__init__()
        self.pool1 = nn.MaxPool2d(1)
        self.pool2 = nn.MaxPool2d(2)
        self.pool3 = nn.MaxPool2d(3)
        self.pool4 = nn.MaxPool2d(6)
        self.bottleneck = nn.Conv2d(features * 4, out_features, kernel_size=1)
        self.relu = nn.ReLU()
        
    def forward(self, x):    # x:512 * 64
        p1 = F.interpolate(self.pool1(x), size = [16, 16])    # 512 * 64
        p2 = F.interpolate(self.pool2(x), size = [16, 16])    # 256 * 64
        p3 = F.interpolate(self.pool3(x), size = [16, 16])    # 170 * 64
        p4 = F.interpolate(self.pool4(x), size = [16, 16])    # 85 * 64
        x = self.bottleneck(torch.cat([p1, p2, p3, p4], 1))
        return self.relu(x)