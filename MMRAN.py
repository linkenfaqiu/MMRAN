import torch
from torch import nn
import torch.nn.functional as F
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 以下为无1×1卷积代码
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),  # 已添加BN层
            # nn.GroupNorm(64, out_ch),     # 在Batchsize比较小的时候，使用GN层替代BN层可以提升一定的模型精度
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            # nn.GroupNorm(64, out_ch),
            nn.ReLU(inplace=True)
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch)
            # nn.GroupNorm(64, out_ch),
        )

    def forward(self, input):
        out = self.conv(input)
        out = out + self.shortcut(input)
        out = F.relu(out)
        return out

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

# 多尺度卷积模块
class MultiScaleModule(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleModule, self).__init__()
        # 动态调整每个分支的通道数
        branch_channels = in_channels // 4
        if in_channels % 4 != 0:
            raise ValueError(f"in_channels ({in_channels}) must be divisible by 4 for MultiScaleModule.")

        self.conv0 = nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, bias=False)  # 1x1卷积
        self.conv1 = nn.Conv2d(in_channels, branch_channels, kernel_size=3, padding=1, bias=False)  # 3x3卷积
        self.conv2 = nn.Conv2d(in_channels, branch_channels, kernel_size=5, padding=2, bias=False)  # 5x5卷积
        self.conv3 = nn.Conv2d(in_channels, branch_channels, kernel_size=7, padding=3, bias=False)  # 7x7卷积
        self.norm = nn.BatchNorm2d(in_channels)  # 对最终结果进行归一化

    def forward(self, x):
        # 四个并行卷积
        F0 = self.conv0(x)  # 1x1卷积
        F1 = self.conv1(x)  # 3x3卷积
        F2 = self.conv2(x)  # 5x5卷积
        F3 = self.conv3(x)  # 7x7卷积
        # 通道维度拼接 F0, F1, F2, F3
        F_out = torch.cat([F0, F1, F2, F3], dim=1)  # [B, C, H, W]
        F_out = self.norm(F_out)  # 归一化
        return F_out

# scSE模块，结合MultiScaleModule
class MRAM(nn.Module):
    def __init__(self, in_channels):
        super(MRAM, self).__init__()
        self.multi_scale = MultiScaleModule(in_channels)  # 添加多尺度卷积模块
        self.cSE = cSE(in_channels)
        self.sSE = sSE(in_channels)

    def forward(self, U):
        U = self.multi_scale(U)  # 先经过多尺度卷积模块
        U_cse = self.cSE(U)  # 通道注意力
        U_sse = self.sSE(U_cse)  # 空间注意力
        return U_sse + U  # 残差连接
      
      
class MMRAN(nn.Module):
    def __init__(self, in_ch, out_ch, reduction_factor=4):
        super(MMRAN, self).__init__()
        # 确保 reduction_factor 只能取值 1, 2, 4
        if reduction_factor not in [1, 2, 4]:
            raise ValueError(f"Invalid reduction_factor: {reduction_factor}. It must be 1, 2, or 4.")
        factor = reduction_factor
        print(f"Factor={factor} (default=4), all channels of Convolutional Layers will be reduced to 1 / {factor}.")
        
        # 通道数根据 factor 调整
        self.conv1 = DoubleConv(in_ch, 64 // factor)  # 原 64
        self.conv2 = DoubleConv(64 // factor, 128 // factor)  # 原 128
        self.conv3 = DoubleConv(128 // factor, 256 // factor)  # 原 256
        self.conv4 = DoubleConv(256 // factor, 512 // factor)  # 原 512
        self.conv5 = DoubleConv(512 // factor, 1024 // factor)  # 原 512

        self.pool = nn.MaxPool2d(2)  # 共享池化层

        # 上采样分支
        self.up6 = nn.ConvTranspose2d(1024 // factor, 512 // factor, 2, stride=2)  # 原 1024->512
        self.conv6 = DoubleConv(1024 // factor, 512 // factor)  # 原 1024->512

        self.up7 = nn.ConvTranspose2d(512 // factor, 256 // factor, 2, stride=2)  # 原 512->256
        self.conv7 = DoubleConv(512 // factor, 256 // factor)  # 原 512->256

        self.up8 = nn.ConvTranspose2d(256 // factor, 128 // factor, 2, stride=2)  # 原 256->128
        self.conv8 = DoubleConv(256 // factor, 128 // factor)  # 原 256->128

        self.up9 = nn.ConvTranspose2d(128 // factor, 64 // factor, 2, stride=2)  # 原 128->64
        self.conv9 = DoubleConv(128 // factor, 64 // factor)  # 原 128->64

        self.conv10 = nn.Conv2d(64 // factor, out_ch, 1)  # 输出通道数不变

        self.num_levels = 4
        self.pool_type = 'max_pool'

        # 下采样分支
        self.conv11 = DoubleConv(1024 // factor, 512 // factor)  # 原 1024->512
        self.conv12 = DoubleConv(512 // factor, 256 // factor)  # 原 512->256
        self.conv13 = DoubleConv(256 // factor, 128 // factor)  # 原 256->128
        self.conv14 = DoubleConv(128 // factor, 64 // factor)  # 原 128->64

        self.fc1 = nn.Linear(1920 // factor, 100)  # 原 1920，减半
        self.fc2 = nn.Linear(100, 3)        # 3分类

        self.c_se1 = MRAM(64 // factor)
        self.c_se2 = MRAM(128 // factor)
        self.c_se3 = MRAM(256 // factor)
        self.c_se4 = MRAM(512 // factor)

    def forward(self, x):
        x = self.conv1(x)  # 512 * 512 * (32/64)
        att1 = self.c_se1(x)
        x = self.pool(x)  # 256 * 256 * (32/64)

        x = self.conv2(x)  # 256 * 256 * (64/128)
        att2 = self.c_se2(x)
        x = self.pool(x)  # 128 * 128 * (64/128)

        x = self.conv3(x)  # 128 * 128 * (128/256)
        att3 = self.c_se3(x)
        x = self.pool(x)  # 64 * 64 * (128/256)

        x = self.conv4(x)  # 64 * 64 * (256/512)
        att4 = self.c_se4(x)
        x = self.pool(x)  # 32 * 32 * (256/512)

        x = self.conv5(x)  # 32 * 32 * (256/512)\
            
        # 在本文中并没有使用这个模块，但是您也可以加上以提升性能
        # x = self.psp(x)      #在网络最底层增加了多尺度融合

        # 上采样部分
        x_up = self.up6(x)  # 64 * 64 * (256/512)
        x_up = torch.cat([x_up, att4], dim=1)  # 64 * 64 * (512/1024)
        x_up = self.conv6(x_up)  # 64 * 64 * (512/1024)

        x_up = self.up7(x_up)  # 128 * 128 * (128/256)
        x_up = torch.cat([x_up, att3], dim=1)  # 128 * 128 * (256/512)
        x_up = self.conv7(x_up)  # 128 * 128 * (128/256)

        x_up = self.up8(x_up)  # 256 * 256 * (64/128)
        x_up = torch.cat([x_up, att2], dim=1)  # 256 * 256 * (128/256)
        x_up = self.conv8(x_up)  # 256 * 256 * (64/128)

        x_up = self.up9(x_up)  # 512 * 512 * (32/64)
        x_up = torch.cat([x_up, att1], dim=1)  # 512 * 512 * (64/128)
        x_up = self.conv9(x_up)  # 512 * 512 * (32/64)

        seg_output = self.conv10(x_up)  # 512 * 512 * out_ch

        # CNN部分
        x = self.conv11(x)  # 32 * 32 * (256/512)
        x = self.pool(x)  # 16 * 16 * (256/512)
        x = self.conv12(x)  # 16 * 16 * (128/256)
        
        # 在本文中并没有使用这个模块，但是您也可以加上以提升性能
        # x = self.psp2(x)
        x = self.pool(x)  # 8 * 8 * (128/256)
        x = self.conv13(x)  # 8 * 8 * (64/128)
        x = self.pool(x)  # 4 * 4 * (64/128)
        x = self.conv14(x)  # 4 * 4 * (32/64)

        # SPP 层
        spp_layer = SPPLayer(self.num_levels, self.pool_type)
        x = spp_layer(x)

        x = F.relu(self.fc1(x))
        cls_output = self.fc2(x)

        return seg_output, cls_output


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
            print("Focal_loss alpha = {}, Fine tune the assignment of weights for each category".format(alpha))
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1  # 如果α为一个常数,则降低第一类的影响,在目标检测中为第一类
            print(" --- Focal_loss alpha = {} --- ".format(alpha))
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
      
          
class SPPLayer(torch.nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type
        
    def forward(self, x):
        # num:样本数量 c:通道数 h:高 w:宽
        # num: the number of samples
        # c: the number of channels
        # h: height
        # w: width
        num, c, h, w = x.size() 
        # print(x.size())
        for i in range(self.num_levels):
            level = i+1

            '''
            The equation is explained on the following site:
            http://www.cnblogs.com/marsggbo/p/8572846.html#autoid-0-0-0
            '''
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.floor(h / level), math.floor(w / level))
            pooling = (math.floor((kernel_size[0]*level-h+1)/2), math.floor((kernel_size[1]*level-w+1)/2))
            
            # update input data with padding
            zero_pad = torch.nn.ZeroPad2d((pooling[1],pooling[1],pooling[0],pooling[0]))
            x_new = zero_pad(x)
            
            # update kernel and stride
            h_new = 2*pooling[0] + h
            w_new = 2*pooling[1] + w
            
            kernel_size = (math.ceil(h_new / level), math.ceil(w_new / level))
            stride = (math.floor(h_new / level), math.floor(w_new / level))
            
            
            # 选择池化方式 
            if self.pool_type == 'max_pool':
                try:
                    tensor = F.max_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
                except Exception as e:
                    print(str(e))
                    print(x.size())
                    print(level)
            else:
                tensor = F.avg_pool2d(x_new, kernel_size=kernel_size, stride=stride).view(num, -1)
            
            # 展开、拼接
            if (i == 0):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten
      
      
 # PSP模块，以下两个模块在文中并没有用到，但是您也可以在网络中使用它们，对分类效果的提升有一定的帮助。
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
