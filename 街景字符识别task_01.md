# 街景字符识别task_01

### 赛题介绍

​    天池大数据竞赛：https://tianchi.aliyun.com/competition/entrance/531795/introduction?spm=5176.12281949.1003.1.493e24482oqmKQ。

​    零基础入门CV街道字符识别，提高数据建模能力，本赛题以计算机字符识别为背景，预测介绍字符编码，赛题数据采用公开数据集SVHN。

#### 数据处理

数据集下载地址：https://tianchi.aliyun.com/competition/entrance/531795/information

或者http://ufldl.stanford.edu/housenumbers/，数据集均已进行匿名处理和噪音处理

所有的数据的标注使用json格式，并使用文件名进行索引。字符位置具体如下：

| filed  | description          |      |
| ------ | -------------------- | ---- |
| top    | 左上角x坐标          |      |
| height | 字符高度h            |      |
| left   | 左上角y坐标          |      |
| width  | 字符宽w              |      |
| label  | 字符编码（具体分类） |      |

训练集包括3W张图片，验证集数据包括1W张图片。

下面是其字符坐标：

![image-20200518190926323](C:\Users\zst\AppData\Roaming\Typora\typora-user-images\image-20200518190926323.png)

在数据集中，同一个照片可能会出现多个字符 ，因此，在比赛数据的JSON标注中，会有两个字符的边框信息，如下图所示：

![image-20200518191041230](C:\Users\zst\AppData\Roaming\Typora\typora-user-images\image-20200518191041230.png)

#### 读取数据

```python
import json
import  numpy as np
import cv2
import  matplotlib.pyplot as plt
#加载数据，这是下载下来的的json文件，根据自己的目录读入
train_json = json.load(open('D:\cv_project\str_recognition/train.json'))

# 数据标注处理
def parse_json(d):
   #讲标注数据放入数组里
   arr = np.array([
       d['top'], d['height'], d['left'],  d['width'], d['label']
   ])
   #转化为int形
   arr = arr.astype(int)
  # print(arr)  可以用来查看arr里面具体形式
   return arr

#读入图片 以000000.jpg为例
img = cv2.imread('D:\cv_project\dataset\shvm/train/000000.png')
arr = parse_json(train_json['000000.png'])#调用上面数据标注处理函数

#底色设置  具体可以查看figure函数用法
plt.figure(figsize=(10, 10))
# subplot函数 各个参数分别代表  行 列  第几幅图
plt.subplot(1, arr.shape[1]+1, 1)
#print(arr.shape[1])   可以查看arr里面有几个维度
plt.imshow(img)#将图片显示
plt.show()#如果是pycharm编译器  得加show才能show出图片
plt.xticks([]); plt.yticks([])  #设置xy轴

#对标签维度进行循环 
for idx in range(arr.shape[1]):
   plt.subplot(1, arr.shape[1]+1, idx+2)  #行 列 第几副图 
   #print(idx)
   # 切片  图像的两个部分显示出来   根据arr的内容
#相当于 先画出 左边框线长  再画出宽  下底面。。    
   plt.imshow(img[arr[0, idx]:arr[0, idx]+arr[1, idx],arr[2, idx]:arr[2, idx]+arr[3, idx]]) #第一个标注位置x  到第一个标注位置结束     第二个标注位置  第二个标注为结束 
   #print(arr[1,idx])
   plt.show()
   #符号隶属于哪一类 
   plt.title(arr[4, idx])
   plt.xticks([]); plt.yticks([])#横纵坐标 

```

原图：![image-20200518193624545](C:\Users\zst\AppData\Roaming\Typora\typora-user-images\image-20200518193624545.png)

切片图：![image-20200518193658140](C:\Users\zst\AppData\Roaming\Typora\typora-user-images\image-20200518193658140.png)

图2：![image-20200518193727676](C:\Users\zst\AppData\Roaming\Typora\typora-user-images\image-20200518193727676.png)

#### 解题思路

因为每个数据可能含有的字符数不唯一，与传统的分类任务不同。

##### 思路一：

改变数据成为一个固定字符数的识别问题，比如最长的字符个数为6，那么将不足六位的字符数补齐，如1XXXXXX,12XXXX,123XXX.......这样做的好处就是可以简化成为6个字符的分类问题，再对这些字符进行分类，如果是填充字符，则表示该字符为空。

##### 思路2：

参考典型的CRNN字符识别模型，把字符看做是一个单词或者句子

##### 思路3：

先进行定位，再对定位框里面的字符进行识别，利用目标检测的想法。如YOLO、SSD等。

三种方法难度从低到高，笔者采用YOLOV3对字符进行识别。



