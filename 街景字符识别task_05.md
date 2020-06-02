- 在训练集上进行训练，并在验证集上进行验证；
- 模型可以保存最优的权重，并读取权重；
- 记录下训练集和验证集的精度，便于调参。

##  模型训练与验证

### 学习目标

- 理解验证集的作用，并使用训练集和验证集完成训练
- 学会使用Pytorch环境下的模型读取和加载，并了解调参流程

### 构造验证集

- #### 训练集（Train Set）：模型用于训练和调整模型参数；

- #### 验证集（Validation Set）：用来验证模型精度和调整模型超参数；

- #### 测试集（Test Set）：验证模型的泛化能力。

因为训练集和验证集是分开的，所以模型在验证集上面的精度在一定程度上可以反映模型的泛化能力。在划分验证集的时候，需要注意验证集的分布应该与测试集尽量保持一致，不然模型在验证集上的精度就失去了指导意义。

- #### 留出法（Hold-Out）

直接将训练集划分成两部分，新的训练集和验证集。这种划分方式的优点是最为直接简单；缺点是只得到了一份验证集，有可能导致模型在验证集上过拟合。留出法应用场景是数据量比较大的情况。

- #### 交叉验证法（Cross Validation，CV）

将训练集划分成K份，将其中的K-1份作为训练集，剩余的1份作为验证集，循环K训练。这种划分方式是所有的训练集都是验证集，最终模型验证精度是K份平均得到。这种方式的优点是验证集精度比较可靠，训练K次可以得到K个有多样性差异的模型；CV验证的缺点是需要训练K次，不适合数据量很大的情况。

- #### 自助采样法（BootStrap）

通过有放回的采样方式得到新的训练集和验证集，每次的训练集和验证集都是有区别的。这种划分方式一般适用于数据量较小的情况。

### 集成学习方法

在机器学习中的集成学习可以在一定程度上提高预测精度，常见的集成学习方法有Stacking、Bagging和Boosting，同时这些集成学习方法与具体验证集划分联系紧密。

由于深度学习模型一般需要较长的训练周期，如果硬件设备不允许建议选取留出法，如果需要追求精度可以使用交叉验证的方法。

### 深度学习中的集成学习

#### Dropout

Dropout可以作为训练深度神经网络的一种技巧。在每个训练批次中，通过随机让一部分的节点停止工作。同时在预测的过程中让所有的节点都其作用。

Dropout经常出现在在先有的CNN网络中，可以有效的缓解模型过拟合的情况，也可以在预测时增加模型的精度。

加入Dropout后的网络结构如下:

```python
# 定义模型
class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()
        # CNN提取特征模块
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(), 
            nn.Dropout(0.25),
            nn.MaxPool2d(2),
        )
        # 
        self.fc1 = nn.Linear(32*3*7, 11)
        self.fc2 = nn.Linear(32*3*7, 11)
        self.fc3 = nn.Linear(32*3*7, 11)
        self.fc4 = nn.Linear(32*3*7, 11)
        self.fc5 = nn.Linear(32*3*7, 11)
        self.fc6 = nn.Linear(32*3*7, 11)
    
    def forward(self, img):        
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        c6 = self.fc6(feat)
        return c1, c2, c3, c4, c5, c6
```

#### TTA

测试集数据扩增（Test Time Augmentation，简称TTA）也是常用的集成学习技巧，数据扩增不仅可以在训练时候用，而且可以同样在预测时候进行数据扩增，对同一个样本预测三次，然后对三次结果进行平均。

```python
def predict(test_loader, model, tta=10):
   model.eval()
   test_pred_tta = None
   # TTA 次数
   for _ in range(tta):
       test_pred = []
   
       with torch.no_grad():
           for i, (input, target) in enumerate(test_loader):
               c0, c1, c2, c3, c4, c5 = model(data[0])
               output = np.concatenate([c0.data.numpy(), c1.data.numpy(),
                  c2.data.numpy(), c3.data.numpy(),
                  c4.data.numpy(), c5.data.numpy()], axis=1)
               test_pred.append(output)
       
       test_pred = np.vstack(test_pred)
       if test_pred_tta is None:
           test_pred_tta = test_pred
       else:
           test_pred_tta += test_pred
   
   return test_pred_tta
```

#### Snapshot

在论文Snapshot Ensembles中，作者提出使用cyclical learning rate进行训练模型，并保存精度比较好的一些checkopint，最后将多个checkpoint进行模型集成。

由于在cyclical learning rate中学习率的变化有周期性变大和减少的行为，因此CNN模型很有可能在跳出局部最优进入另一个局部最优。在Snapshot论文中作者通过使用表明，此种方法可以在一定程度上提高模型精度，但需要更长的训练时间。

训练20个epoch后：

红线：训练集loss

蓝线：验证集loss

绿线：训练集精度

蓝点：验证集精度

![image-20200602233935782](C:\Users\zst\AppData\Roaming\Typora\typora-user-images\image-20200602233935782.png)

字符精度：

![image-20200602233904464](C:\Users\zst\AppData\Roaming\Typora\typora-user-images\image-20200602233904464.png)