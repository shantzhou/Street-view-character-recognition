



## 数据读取与数据增广

#### 目标：

- 学会pytorch和pytorch中图像读取
- 学会增广方法，利用pytorch读取赛题的数据

#### 常用的torchvision数据增强方法

- transforms.CenterCrop 对图片中心进行裁剪
- transforms.ColorJitter 对图像颜色的对比度、饱和度和零度进行变换
- transforms.FiveCrop 对图像四个角和中心进行裁剪得到五分图像
- transforms.Grayscale 对图像进行灰度变换
- transforms.Pad 使用固定值进行像素填充
- transforms.RandomAffine 随机仿射变换
- transforms.RandomCrop 随机区域裁剪
- transforms.RandomHorizontalFlip 随机水平翻转
- transforms.RandomRotation 随机旋转
- transforms.RandomVerticalFlip 随机垂直翻转

###### note：本次采用的数据集是字符形式，不能对其翻转操作，如对6进行翻转会变成9

##### 常用数据增强库：

- ##### torchvision    

https://github.com/pytorch/vision   

pytorch官方提供的数据扩增库，提供了基本的数据数据扩增方法，可以无缝与torch进行集成；但数据扩增方法种类较少，且速度中等；

- ##### imgaug

https://github.com/aleju/imgaug
imgaug是常用的第三方数据扩增库，提供了多样的数据扩增方法，且组合起来非常方便，速度较快；

- ##### albumentations

[https://albumentations.readthedocs.io](https://albumentations.readthedocs.io/)
是常用的第三方数据扩增库，提供了多样的数据扩增方法，对图像分类、语义分割、物体检测和关键点检测都支持，速度较快。

## Pytorch读取数据

由于本次赛题我们使用Pytorch框架讲解具体的解决方案，接下来将是解决赛题的第一步使用Pytorch读取赛题数据。
在Pytorch中数据是通过Dataset进行封装，并通过DataLoder进行并行读取。所以我们只需要重载一下数据读取的逻辑就可以完成数据的读取。

- ##### Dataset：对数据集的封装，提供索引方式的对数据样本进行读取

- ##### DataLoder：对Dataset进行封装，提供批量读取的迭代读取

```python
class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        #打开转换为rgb
        img = Image.open(self.img_path[index]).convert('RGB')
        #transform不为None  就去做transform
        if self.transform is not None:
            img = self.transform(img)

        # 原始SVHN中类别10为数字0,这里的10值得是空字符 实现上节所讲的  2xxxxx 12xxxx 123xxx
        lbl = np.array(self.img_label[index], dtype=np.int)
        lb1 = 1
        lbl = list(lbl) + (6 - len(lbl)) * [10]

        return img, torch.from_numpy(np.array(lbl[:6]))

    def __len__(self):
        return len(self.img_path)
train_path = glob.glob('train/*.png')
train_path.sort()
train_json = json.load(open('/train.json'))
train_label = [train_json[x]['label'] for x in train_json]

data = SVHNDataset(train_path, train_label,
          transforms.Compose([#将各个变化串联在一起
              # 缩放到固定尺寸
              transforms.Resize((64, 128)),
              # 随机颜色变换
              transforms.ColorJitter(0.2, 0.2, 0.2),
              # 加入随机旋转
              transforms.RandomRotation(5),

              # 将图片转换为pytorch 的tesntor
              # transforms.ToTensor(),

              # 对图像像素进行归一化
              # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ]))
```

加入DataLoder后，数据读取代码改为如下：

##### 新建dataset类要继承（from torch.utils.data.dataset import Dataset）类，

##### 并重写__init__，**getitem**，__len__函数

```python
class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        #打开转换为rgb
        img = Image.open(self.img_path[index]).convert('RGB')
        #transform不为None  就去做transform
        if self.transform is not None:
            img = self.transform(img)

        # 原始SVHN中类别10为数字0,这里的10值得是空字符 实现上节所讲的  2xxxxx 12xxxx 123xxx
        lbl = np.array(self.img_label[index], dtype=np.int)
        lbl = list(lbl) + (6 - len(lbl)) * [10]

        return img, torch.from_numpy(np.array(lbl[:6]))

    def __len__(self):
        return len(self.img_path)


train_path = glob.glob('D:\cv_project\dataset\shvm/train/*.png')
train_path.sort()
train_json = json.load(open('D:\cv_project\str_recognition/train.json'))
train_label = [train_json[x]['label'] for x in train_json]
#继承dataloader
train_loader =torch.utils.data.DataLoader(
     SVHNDataset(train_path, train_label,
                   transforms.Compose([# 串联多个图片的操作
                       # 缩放到固定尺寸
                       transforms.Resize((64, 128)),
                       # 随机颜色变换
                       transforms.ColorJitter(0.2, 0.2, 0.2),
                       # 加入随机旋转
                       transforms.RandomRotation(5),
                       # 将图片转换为pytorch 的tesntor
                       transforms.ToTensor(),
                       # 对图像像素进行归一化
                       transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
                   ])),
    batch_size=8,#每次送入图片数目
    shuffle=True,#打乱排序
    num_workers=0)#线程数目

#也可以构造iter对象，然后通过next函数一个一个看。
x = iter(train_loader)
data = next(x)
#data0 是图像的数据   data1 是lebel的数据  
img, label = data[0], data[1]
print(label)
#格式为batchsize  channel  h w 
print(img.size())   #torch.Size([4, 3, 64, 128])
#格式为 batchsize   label数量长度
print(label.size())     #torch.Size([4, 6])

```

##### opencv数据增强

```python
import cv2

img = cv2.imread('dataset/000041.png')
#缩放尺寸

#注意.resize()的第二个参数表示目标尺寸，分别为(w, h)，而.imread()读入的是按照(h, w, c),注意区别。
resized_img = cv2.resize(img, (128, 64))     #(W, H)
print(img.shape)        #原图尺寸：(114, 287, 3)分别为(H, W, C)
print(resize_img.shape)     #(64, 128, 3)分别为(H, W, C)

cv2.imshow('img',img)
cv2.imshow('resized_img', resized_img)
cv2.waitKey()

#随机颜色变换
# OpenCV中亮度和对比度公式：g(x) = αf(x) + β，其中：α(>0)、β常称为增益与偏置值，分别控制图片的对比度和亮度。
res1_img = np.uint8(np.clip((0.1*img+10), 0, 255))
res2_img = np.uint8(np.clip((2*img+10), 0, 255))
cv2.imshow('img',img)
cv2.imshow('darken', res1_img)
cv2.imshow('lighten', res2_img)
cv2.waitKey()

#随机旋转
h, w, c = img.shape
angle = random.randint(0, 360)      #设置随机旋转角度
rotated_mat = cv2.getRotationMatrix2D((w*0.5, h*0.5), angle, 1)     #获取旋转变化矩阵，参数(旋转中心点坐标，旋转角度，缩放因子)
rotated_img = cv2.warpAffine(img, rotated_mat, (w,h))       #参数(原图，旋转矩阵，输出图像的尺寸)
cv2.imshow('img', img)
cv2.imshow('rotated_img', rotated_img)
cv2.waitKey()
```



##### pillow数据增强

```python
from PIL import Image
import random

#resize
img = Image.open('dataset/000041.png')
print(img.size)     #(w, h)没有c
print(img.mode)     #RGB,而opencv读取出是BGR，注意区别
resized_img = img.resize((128, 64))     #(w, h)
img.show()
resized_img.show()

#亮度对比度等调节
from PIL import ImageEnhance
#对比度
contrast_img = ImageEnhance.Contrast(img)
contrast_img.enhance(1.8).show('180%enhance Contrast')	#对比度增强1.8倍
#亮度
lighten_img = ImageEnhance.Brightness(img)
lighten_img.enhance(1.8).show('180%enchance brightness')	#亮度增强1.8倍

#随机旋转
angle = random.randint(0,360)
rotated_img = img.rotate(angle)
rotated_img.show()
```



##### torchvision数据增强

```python
import torch
import torchvision
from torch.utils.data.dataset import Dataset
from PIL import Image
import random
import cv2

angle = random.randint(0, 30)
class Data(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        self.transform = transform

    def __getitem__(self, item):
        img = Image.open(self.img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_path)

new_img = Data(img_path='dataset/000041.png', transform=torchvision.transforms.Compose([                            torchvision.transforms.Resize((64, 128)),
                 torchvision.transforms.ColorJitter(0.2, 0.2, 0.2),      #分别是亮度、对比度、饱和度
                 torchvision.transforms.RandomRotation(angle)]))#随机旋转角度

new_img[0].show()
```

