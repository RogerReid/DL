from pathlib import Path

import numpy as np
import torch
import torchvision.utils
import matplotlib.pyplot as plt
# from torch import classes
from torch.autograd import Variable
from torchvision import datasets, models
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


batch_size = 4
num_workers = 0
classes = ["cat", "dog"]
PATH1 = Path("../input/dogs-vs-cats/test1")
test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.255])])
test_data = datasets.ImageFolder(PATH1, transform=test_transforms)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)


def imageshow(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1 , 2 , 0)))
    plt.show()


def testBatch():
    images, labels = next(iter(test_loader))
    print(labels)

     # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))

    # show the real labels on the screen
    print("Real labels: "," ".join("%5s" % classes[labels[j]] for j in range(batch_size)))

    # Let see what if the model identifiers the labels of those example
    vgg_19 = models.vgg19_bn(pretrained=False)  # 已经进行了预先训练
    path = "model_vgg19_2.pth"
    vgg_19.load_state_dict(torch.load(path), strict=False)
    outputs = vgg_19(images)

    # we got the probability for every labelsm.The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, dim=1)

    # Let s show the predicted labels on the screen to compare with the real ones
    print("Predicted : "," ".join("%5s" % classes[predicted[j]] for j in range(batch_size)))


testBatch()





