## CNN

卷积神经网络（简称CNN）是一类特殊的人工神经网络，是深度学习中重要的一个分支。CNN在很多领域都表现优异，精度和速度比传统计算学习算法高很多。特别是在计算机视觉领域，CNN是解决图像分类、图像检索、物体检测和语义分割的主流模型。

CNN每一层由众多的卷积核组成，每个卷积核对输入的像素进行卷积操作，得到下一次的输入。随着网络层的增加卷积核会逐渐扩大感受野，并缩减图像的尺寸



![image-20200522225654951](C:\Users\zst\AppData\Roaming\Typora\typora-user-images\image-20200522225654951.png)

CNN是一种层次模型，输入的是原始的像素数据。CNN通过卷积（convolution）、池化（pooling）、非线性激活函数（non-linear activation function）和全连接层（fully connected layer）构成。

如下图所示为LeNet网络结构，是非常经典的字符识别模型。两个卷积层，两个池化层，两个全连接层组成。卷积核都是5×5，stride=1，池化层使用最大池化。

![image-20200522225714769](C:\Users\zst\AppData\Roaming\Typora\typora-user-images\image-20200522225714769.png)

通过多次卷积和池化，CNN的最后一层将输入的图像像素映射为具体的输出。如在分类任务中会转换为不同类别的概率输出，然后计算真实标签与CNN模型的预测结果的差异，并通过反向传播更新每层的参数，并在更新完成后再次前向传播，如此反复直到训练完成 。

与传统机器学习模型相比，CNN具有一种端到端（End to End）的思路。在CNN训练的过程中是直接从图像像素到最终的输出，并不涉及到具体的特征提取和构建模型的过程，也不需要人工的参与。

### 常见的CNN：

#### VGG16:



![image-20200522225808671](C:\Users\zst\AppData\Roaming\Typora\typora-user-images\image-20200522225808671.png)

#### ResNet50：

![image-20200522225839404](C:\Users\zst\AppData\Roaming\Typora\typora-user-images\image-20200522225839404.png)



### Pytorch构建CNN模型

```python
# 定义模型
class SVHN_Model1(nn.Module):  #继承module 类
    def __init__(self):
        super(SVHN_Model1, self).__init__()
        # CNN提取特征模块
        self.cnn = nn.Sequential(
            #输入通道数  输出通道数  kernel_size   步长
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2)),  #
            nn.ReLU(),#激活函数
            nn.MaxPool2d(2),  #池化
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2)),#
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        #nn.Linear（）是用于设置网络中的全连接层的，需要注意的是全连接层的输入与输出都是二维张量，一般形状为[batch_size, size]
        #输入张量大小= 输出32深度  3*7是输入图片  输出张量大小：11是指的0-9还有x这个字符
        self.fc1 = nn.Linear(32 * 3 * 7, 11)
        self.fc2 = nn.Linear(32 * 3 * 7, 11)
        self.fc3 = nn.Linear(32 * 3 * 7, 11)
        self.fc4 = nn.Linear(32 * 3 * 7, 11)
        self.fc5 = nn.Linear(32 * 3 * 7, 11)
        self.fc6 = nn.Linear(32 * 3 * 7, 11)

    def forward(self, img):
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1) #输出维度 featshape[0] * -1（自动判定这个值）
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        c6 = self.fc6(feat)
        return c1, c2, c3, c4, c5, c6


model = SVHN_Model1()
```

ps：最终未选择自己搭建的模型，采用resnet18作为训练模型。

```python
class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()
		#未下载则会自动下载
        model_conv = models.resnet18(pretrained=True)
        #将最后的池化改成全局池化 
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv
		#全连接
        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)
        self.fc5 = nn.Linear(512, 11)

#前向传播
    def forward(self, img):
        feat = self.cnn(img)
        # print(feat.shape)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        return c1, c2, c3, c4, c5
```



### 训练过程

```python
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器0.001的学习率
optimizer = torch.optim.Adam(model.parameters(), 0.001)

loss_plot, c0_plot = [], []

# 迭代10个Epoch
for epoch in range(10):
    for data in train_loader:
        c0, c1, c2, c3, c4, c5 = model(data[0])
        #data[1]指的是label  [:,0]取第二列所有]
        loss = criterion(c0, data[1][:, 0]) + \
               criterion(c1, data[1][:, 1]) + \
               criterion(c2, data[1][:, 2]) + \
               criterion(c3, data[1][:, 3]) + \
               criterion(c4, data[1][:, 4]) + \
               criterion(c5, data[1][:, 5])
        loss /= 6
        # 因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和
        optimizer.zero_grad()  #把梯度置零，也就是把loss关于weight的导数变成0

        loss.backward()
        # 即反向传播求梯度
        optimizer.step()
		#画loss
        loss_plot.append(loss.item())  #items()方法把字典中每对key和value组成一个元组，并把这些元组放在列表中返回。
        #np.argmax(a,axis=1) //当axis=1时，表示返回行方向上数值最大值下标
        #画准确度
         loss /= 6
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #
            acc = (c0.argmax(1) == label[:, 0]).sum().item() * 1.0 / c0.shape[0]
            c0_plot.append(acc)
            print('Epoch: {},iter:{} Train loss: {}'.format(epoch, iter, loss))
        # writer.add_scalar("loss", loss / ( + 1), epoch)
    print('结束训练')
    return loss_plot, c0_plot


if __name__ == '__main__':
    #writer = SummaryWriter()
    model = SVHN_Model1()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    loss_plot, c0_plot = train(train_loader, model, criterion=criterion, optimizer=optimizer, epoches= 1,device=device)
#可视化
    plt.figure()
    plt.plot(range(len(loss_plot)), loss_plot, 'r-', linewidth=2)
    plt.title('Loss', fontsize=30)
    plt.legend()
    plt.xlabel('Iterations', fontsize=20)
    plt.ylabel('loss', fontsize=20)

    plt.figure()
    plt.plot(range(len(c0_plot)), c0_plot, 'b-', linewidth=2)
    plt.title('ACC', fontsize=30)
    plt.legend()
    plt.xlabel('Iterations', fontsize=20)
    plt.ylabel('acc', fontsize=20)
    plt.show()
    
```

训练结果可视化：

Loss：

![image-20200526225500999](C:\Users\zst\AppData\Roaming\Typora\typora-user-images\image-20200526225500999.png)

准确度acc：

![image-20200526225512714](C:\Users\zst\AppData\Roaming\Typora\typora-user-images\image-20200526225512714.png)

