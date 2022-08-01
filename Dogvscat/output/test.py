from pathlib import Path

import torch
import torchvision.utils
from matplotlib.pyplot import imshow

from torch.autograd import Variable
from torchvision import datasets, models
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


classes = ["cat", "dog"]
batch_size = 4
num_workers = 0
PATH1 = Path("../input/dogs-vs-cats/test1")
test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.255])])
test_data = datasets.ImageFolder(PATH1, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

dataiter = iter(test_loader)
images, labels = dataiter.next()   # 返回一个batch_size的图片 4张

vgg_19 = models.vgg19_bn(pretrained=True)  # 已经进行了预先训练
check = torch.load("E:\Code\Dogvscat\output\model_vgg19_2.pth")
vgg_19.load_state_dict(check, strict=False)
vgg_19.eval()

"""# 进行总体的正确率计算
correct = 0  # 预测正确的图片数
total = 0 # 总共参与测试的图片数
for data in test_loader:   # 循环每一个batch
    images, labels = data
    outputs = vgg_19(Variable(images))
    _, predicted = torch.max(outputs, dim=1)
    total += labels.size(0)  # 更新测试图片的数量
    correct += (predicted == labels).sum()  # 更新正确分类的图片的数量
print("Accuracy of the network on the test images: %d %%" % ( 100 * correct/ total))"""

# 分类进行计算正确率
class_correct = list(0. for i in range(2))   # 有几个类就弄几个 存储每类中测试正确的个数 列表 舒适化为0
class_total = list(0. for i in range(2))

for data in test_loader:
    images, labels = data
    outputs = vgg_19(Variable(images))
    _, predicted = torch.max(outputs.data, dim=1)
    c = (predicted == labels).squeeze()
    for i in range(4):    # 每个batch都有四个照片，所以要一个4的小循环
        label = labels[i]
        class_correct[label] += c[i]
        class_total[label] += 1
for i in range(2):
    print("Accuracy of %5s : %2d %%" % (classes[i], 100*class_correct[i] / class_total[i]))


