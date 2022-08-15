import SimpleITK
from PIL import Image
from torchvision import transforms
import numpy as np
import os
import nibabel as nib  # nii格式一般都会用到这个包
import imageio  # 转换成图像

# def resize2():
#     filepath = 'D:/study/医学图像处理/unet-liver/u_net_liver/data/val3/'
#     for filename in os.listdir(filepath):
#         img = Image.open(filepath + filename)
#         resize = transforms.Resize([512, 512])
#         img = resize(img)
#         filename = filename.replace('_', '', 1)
#         # filename = filename.replace('.jpg', '.png', 1)
#         printname = str(filename)
#         while printname[0] == "0":
#             printname = printname[1:]
#         print(printname)
#         if printname == '.png':
#             printname = '0.png'
#         img.save('D:/study/医学图像处理/unet-liver/u_net_liver/data/val/' + printname)

def resize(filepath):
    for filename in os.listdir(filepath):
        img = Image.open(filepath + filename)
        # print(img.size[0])
        if (img.size[0] != 512):
            print(filename + "大小不为512" )
            resize = transforms.Resize([512, 512])
            img = resize(img)
            img.save(filepath + filename)

def png_to_nii(png_path):
    mat = []
    mylist = os.listdir(png_path)
    mylist.sort(key = lambda x : int(x.split('.')[0]))     # split：以参数'.'为分隔符，然后取出list第0个元素，即文件名。
    for filename in mylist:
        img1 = Image.open(png_path + filename)
        img2 = np.array(img1)
        mat.append(img2)

    mat = np.array(mat)
    # 此处的mat的格式为样本数*高度*宽度*通道数
    nii_file = SimpleITK.GetImageFromArray(mat)
    SimpleITK.WriteImage(nii_file, png_path + 'test.nii')

def nii_to_png(filepath):
    filenames = os.listdir(filepath)  # 读取nii文件夹
    slice_trans = []
    # cnt = 0
    for f in filenames:
        # 开始读取nii文件
        img_path = os.path.join(filepath, f)
        img = nib.load(img_path)  # 读取nii
        img_fdata = img.get_fdata()
        f = f.replace('.nii', '')  # 去掉nii的后缀名
        # img_f_path = os.path.join(imgfile, fname)
        img_f_path = 'D:/study/dataset/jisuoliu/braintumor_png'
        # 创建nii对应的图像的文件夹
        if not os.path.exists(img_f_path):
            os.mkdir(img_f_path)  # 新建文件夹
        imageio.imwrite(os.path.join(img_f_path, '{}.png'.format(f)), img_fdata)

if __name__ == '__main__':
    # filepath = 'D:/study/dataset/jisuoliu/braintumour_nii'
    # nii_to_image(filepath)

    #nii_to_png_filepath = 'D:/study/dataset/jisuoliu/braintumour_nii'
    #nii_to_png(nii_to_png_filepath)
    filepath = 'D:/study/dataset/jisuoliu/braintumor_png/'
    resize(filepath)