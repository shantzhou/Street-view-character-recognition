## 模型训练与验证

##### 上一讲利用resnet18进行训练，并可视化了训练过程中的误差损失和第一个字符预测准确率

一个合格的深度学习死训练流程包括：

- 在训练集上进行训练，并在验证集上验证
- 模型可以保存最优的权重，并可以读取权重
- 记录训练集和验证集的精度，便于调参

### 构造验证集

#### 过拟合：

在模型的训练过程中，模型只能利用训练数据来进行训练，模型并不能接触到测试集上的样本。因此模型如果将训练集学的过好，模型就会记住训练样本的细节，导致模型在测试集的泛化效果较差

随着模型复杂度和模型训练轮数的增加，CNN模型在训练集上的误差会降低，但在测试集上的误差会逐渐降低，然后逐渐升高，而我们为了追求的是模型在测试集上的精度越高越好。

导致模型过拟合的情况有很多种原因，其中最为常见的情况是模型复杂度（Model Complexity ）太高，导致模型学习到了训练数据的方方面面，学习到了一些细枝末节的规律。

解决上述问题最好的解决方法：构建一个与测试集尽可能分布一致的样本集（可称为验证集），在训练过程中不断验证模型在验证集上的精度，并以此控制模型的训练。

在一般情况下，参赛选手也可以自己在本地划分出一个验证集出来，进行本地验证。训练集、验证集和测试集分别有不同的作用：

- #### 训练集（Train Set）：模型用于训练和调整模型参数；

- #### 验证集（Validation Set）：用来验证模型精度和调整模型超参数；

- #### 测试集（Test Set）：验证模型的泛化能力。

因为训练集和验证集是分开的，所以模型在验证集上面的精度在一定程度上可以反映模型的泛化能力。在划分验证集的时候，需要注意验证集的分布应该与测试集尽量保持一致，不然模型在验证集上的精度就失去了指导意义。



### 模型训练与验证

###### 构造训练集和验证集；

每个epoch的训练结果

```python
def train(train_loader, model, criterion, optimizer,device):
    
    print('开始训练')
    mode1 = model.to(device)
    c0_plot = []
    itera =0
    train_loss = []
    # for epoch in range(epoches):

    for img, label in train_loader:
        itera += 1
        if device:
            img,label = img.to(device),label.to(device)
            label = label.long()

        c0, c1, c2, c3, c4 = model(img)
        loss = criterion(c0, label[:, 0]) + \
               criterion(c1, label[:, 1]) + \
               criterion(c2, label[:, 2]) + \
               criterion(c3, label[:, 3]) + \
               criterion(c4, label[:, 4])

        loss /= 6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # train_loss.append(loss.item())
        train_loss.append(loss.item())
        acc = (c0.argmax(1) == label[:, 0]).sum().item() * 1.0 / c0.shape[0]
        c0_plot.append(acc)
        print('Epoch: {},iter:{} Train loss: {}'.format(epoch, itera, loss))

    # writer.add_scalar("loss", loss / ( + 1), epoch)
    print('结束训练')
    return np.mean(train_loss),itera,c0_plot

```

验证集每个epoch 代码如下：

```python


def validate(val_loader, model, criterion, device):

    val_loss = []
    c1_plot = []
    #不记录梯度信息
    iters =0
    print('开始验证')
    with torch.no_grad():
        for img, label in val_loader:
            iters += 1
            if device:
                img = img.to(device)
                label = label.to(device)
                label =label.long()

            c0, c1, c2, c3, c4 = model(img)

            loss_v = criterion(c0, label[:,0]) + criterion(c1, label[:,1]) + \
                criterion(c2, label[:,2]) + criterion(c3, label[:,3]) + \
                criterion(c4, label[:,4])

            loss_v /= 6
            val_loss.append(loss_v.item())
            print('Epoch: {},iter:{} val_loss: {}'.format(epoch, iters, loss_v))
            accq = (c0.argmax(1) == label[:, 0]).sum().item() * 1.0 / c0.shape[0]
            c1_plot.append(accq)
    print("结束")
    return np.mean(val_loss),c1_plot
```

模型保存：

```python
if val_loss < best_loss:
    best_loss = val_loss
    torch.save(model.state_dict(), './model/model.pt')
```

训练集与验证集效果对比：

```python
if __name__ == '__main__':
    writer = SummaryWriter()
    model = SVHN_Model1()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(torch.cuda.is_available())
    model = model.to(device)
    best_loss = 1000.0
    writer = SummaryWriter()
    train_, valid_ = [], []
    for epoch in range(20):
        train_loss,itera,c0_plot=train(train_loader, model,criterion,optimizer,device=device)
        train_.append(train_loss)
        val_loss,c1_plot = validate(val_loader,model, criterion, device = device)
        valid_.append(val_loss)
         if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), './model/model.pt')
    L1 = plt.plot(range(len(train_)),train_,'r-',valid_,'b-')
    plt.legend(L1)

    plt.xlabel('epoch', fontsize=15)
    plt.ylabel('train VS valid', fontsize=15)
    plt.show()
    print(c0_plot)
    print(c1_plot)
```

训练集与验证集loss图对比：
![image-20200530233842481](C:\Users\zst\AppData\Roaming\Typora\typora-user-images\image-20200530233842481.png)



