import torchvision
import torch
import json
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms
from PIL import Image

import torch.nn as nn
import torchvision.models as models


def save_predictions(student_id, predictions):
    # Please use this function to generate 'XXXXXXX_4_result.json'
    # `predictions` is a list of int (0 or 1; fake=0 and real=1)
    # For example, `predictions[0]` is the prediction given "unknown/0000.jpg".
    # it will be 1 if your model think it is real, else 0 (fake).

    assert isinstance(student_id, str)
    assert isinstance(predictions, list)
    assert len(predictions) == 5350

    for y in predictions:
        assert y in (0, 1)

    with open('{}_4_result.json'.format(student_id), 'w') as f:
        json.dump(predictions, f)


class Config():
    train_dir = 'root'
    unknown_dir = 'unknown'
    epoch = 1
    bc = 64
    gpu = 'True'
    lr = 1e-3


class Mydataset(Dataset):
    def __init__(self, file_list, transform):
        self.file_list = file_list
        self.transform = transform

    def __getitem__(self, index):
        if len(self.file_list[index]) == 2:
            img_path, label = self.file_list[index]
            img = Image.open(img_path)

            if self.transform:
                img = self.transform(img)

            return img, label

        else:
            fn = self.file_list[index]
            img = Image.open(fn).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)

            return img

    def __len__(self):
        return len(self.file_list)

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

train_folder = ImageFolder(root=Config.train_dir)

train_dataset = Mydataset(train_folder.imgs, transform=preprocess)

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=Config.bc, shuffle=True)

net = models.resnet18(pretrained=True)
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, 2)
if Config.gpu:
    net = net.cuda()

optimizer = torch.optim.Adam(net.parameters(), lr=Config.lr)
loss_function = nn.CrossEntropyLoss()

for epoch in range(Config.epoch):
    for step, data in enumerate(train_loader):
        img, label = data
        if Config.gpu:
            img = img.cuda()
            label = label.cuda()

        output = net(img)
        loss = loss_function(output, label)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % 50 == 0:
            print('Epoch:{}, Step:{}, Loss:{:.4f}'.format(
                epoch, step, loss.item()))

i = 0
unknown_list = []
while True:
    if i == 5350:
        break
    length = len(str(i))
    if length != 4:
        unknown_list.append('unknown/'+(4 - length) * '0' + str(i) + '.jpg')
    else:
        unknown_list.append('unknown/'+str(i) + '.jpg')
    i += 1

unknown_data = Mydataset(unknown_list, transform=preprocess)
unknown_loader = DataLoader(
    dataset=unknown_data, batch_size=Config.bc, shuffle=False)

ans = []
with torch.no_grad():
    for step, data in enumerate(unknown_loader):
        data = data.to('cuda')
        output = net(data)
        pred = output.argmax(dim=1, keepdim=True)

        pred = pred.tolist()
        for e in pred:
            ans.extend(e)

save_predictions('0711540', ans)
