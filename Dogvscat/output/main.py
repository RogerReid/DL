from pathlib import Path

import numpy as np

import os

import torch
import torch.nn as nn
from torchvision import datasets, models,transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
from matplotlib import pyplot as plt
import numpy as np


# check id cuda is acailable
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print("cuda is not avaiavle. Training on CPU...")
else:
    print("cuda is available! Training on GPU...")

PATH = Path("../input/dogs-vs-cats-for-pytorch/cat_dog_data/Cat_Dog_data")
PATH1 = Path("../input/dogs-vs-cats/test1")

TRAIN = Path(PATH/"train")
VALID = Path(PATH/"test")
TEST = Path(PATH1)

# number of subprocesses to use for data loading
num_workers = 0
# how many samples per batch to load
batch_size = 32

# convert data to a normalized torch.FloatTensor
train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],   # 将数据转换为标准正态分布 使模型更容易收敛
                                                            [0.229, 0.224, 0.225])])  # 每个照片三个通道 前一个mean 后一个std  image=(image-mean)/std

valid_transforms = transforms.Compose([transforms.Resize((224,224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.299, 0.244, 0.255])])

# chose the training and test datasets
test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.255])])

# chose the training and test datasets
train_data = datasets.ImageFolder(TRAIN, transform=train_transforms)
valid_data = datasets.ImageFolder(VALID, transform=valid_transforms)
test_data = datasets.ImageFolder(PATH1, transform=test_transforms)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data,batch_size=batch_size, num_workers=num_workers,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

images,labels = next(iter(train_loader))  # 返回迭代器的下一个项目  iter（）转化为迭代器对象

classes = ["cat", "dog"]
mean, std = torch.tensor([0.485,0.456,0.406]), torch.tensor([0.229,0,224,0.225])
def denormalize(image):
    image = transforms.Normalize(-mean/std, 1/std)(image)  # denormalize 非规范化
    image = image.permute(1,2,0)  # changing from 3x224x224 to 224x224x3  将维度转换
    image = torch.clamp(image,0,1)  # 将输入 转化为 0 to 1之间 不符合的重新生成一个数
    return image

# helper function to un-normalize and display an image
def imgshow(img):
    img=denormalize(img)
    plt.imshow(img)

# obtain one batch of training images
dataiter = iter(train_loader)
# images,labels = dataiter.next()
# convert images to numpy for display

#plot the images in the batch ,along with the corresponding labels
fig = plt.figure(figsize=(25, 8))

# load the model vgg-19
vgg_19 = models.vgg19_bn(pretrained=True)  # 已经进行了预先训练

# Freeze parameters so we dont backprop through them  backprop反向传播
for parm in vgg_19.parameters():
    parm.requires_grad = False   # 屏蔽与训练模型中的权重，只训练最后一层的全连接的权重

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([    # 有序的字典
    ("fc1", nn.Linear(25088,1028)),
    ("relu1", nn.ReLU()),
    ("dropout1", nn.Dropout(0.5)),
    ("fc2", nn.Linear(1028,512)),
    ("relu2", nn.ReLU()),
    ("dropout2", nn.Dropout(0.5)),
    ("fc3", nn.Linear(512,2)),
    ("output", nn.LogSoftmax(dim=1))
]))
vgg_19.classifier = classifier

criterion = nn.NLLLoss()  # 最大似然损失函数 求标签对应的损失值
optimizer = torch.optim.Adam(vgg_19.parameters(), lr=0.01)  # 优化器

if train_on_gpu:
    vgg_19.cuda()
# number of epochs to train the model
n_epochs = 50
valid_loss_min = np.Inf  # track change in validation loss  无穷大

# train_losses, valid_losses=[],[]
for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0

    # train the model
    vgg_19.train()
    for data,target in train_loader:
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables  明确所有优化变量的梯度
        optimizer.zero_grad()
        # forward pass:compute predicted outputs by passing inputs to the model
        output = vgg_19(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass:compute gradient of the loss with respect to model parameter
        loss.backward()
        # perform as single optimization step(parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()*data.size(0)


    # validate the model  验证集没有优化器 损失函数也不用backward
    vgg_19.eval()
    for data,target in valid_loader:
        # move tensor to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # forward pass : compute predicted outputs by passing input to model 前向传递
        output = vgg_19(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss
        valid_loss += loss.item()*data.size(0)

    # calculate average losses
    # train_losses.append(train_loss/len(train_loader.dataset))
    # valid_losses.append(valid_loss.item()/len(valid_loader.dataset)
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)

    # print training/validation statistics
    print("Epoch:{} \tTraining Loss:{:.6f} \tValidataion Loss: {:.6f}".format(epoch, train_loss, valid_loss))

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print("Validation loss decreased({:.6f} --> {:.6f}. Saving model ...".format(
            valid_loss_min,
            valid_loss)
        )
        torch.save(vgg_19.state_dict(), "model_vgg19_2.pth")  # state_dict保存学习到参数
        valid_loss_min = valid_loss


