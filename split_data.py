from torchvision import datasets
from PIL import Image
import os
import random

split_ratio = 0.8


def mkdir(path):
    # 判斷目錄是否存在
    # 存在：True
    # 不存在：False
    folder = os.path.exists(path)

    # 判斷結果
    if not folder:
        # 如果不存在，則建立新目錄
        os.makedirs(path)
        print('-----建立成功-----')

    else:
        # 如果目錄已存在，則不建立，提示目錄已存在
        print(path+'目錄已存在')


def data_split(full_list, ratio, shuffle=False):
    n_split = int(len(full_list)*ratio)
    if shuffle:
        random.shuffle(full_list)
    return (full_list[:n_split], full_list[n_split:])


datafolder = datasets.ImageFolder(root='fruits-360/Training')
classes = datafolder.classes

paths = {}
for i in range(131):
    paths[i] = []

for info in datafolder.imgs:
    category = info[1]
    path = info[0]
    paths[category].append(path)


for key, path_list in paths.items():
    train_list, valid_list = data_split(path_list, split_ratio, shuffle=True)

    for path in train_list:
        name = path.split('\\')[-1]
        img = Image.open(path)
        new_path = 'Train/'+classes[key]+'/'+name
        mkdir('Train/'+classes[key])
        img.save(new_path)

    for path in valid_list:
        name = path.split('\\')[-1]
        img = Image.open(path)
        new_path = 'Valid/'+classes[key]+'/'+name
        mkdir('Valid/'+classes[key])
        img.save(new_path)
