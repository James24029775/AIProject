import torch
from torchvision import datasets, models, transforms
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import model_method

# Set the train and validation directory paths
train_directory = 'Fruit/Train/'
valid_directory = 'Fruit/Valid/'
test_directory = 'Fruit/Test/'


################################### Model Setting ###################################

# Batch size
bs = 128
# Number of epochs
num_epochs = 1
# Number of classes
num_classes = 131

# Applying transforms to the data
image_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=100, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.CenterCrop(size=87.5),
        # transforms.FiveCrop(196),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=100),
        transforms.CenterCrop(size=87.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(size=100),
        transforms.CenterCrop(size=87.5),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

# Load data from folders
dataset = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid']),
    'test': datasets.ImageFolder(root=test_directory, transform=image_transforms['test'])
}

# Size of train, validation data and test data
dataset_sizes = {
    'train': len(dataset['train']),
    'valid': len(dataset['valid']),
    'test': len(dataset['test'])
}

# Create iterators for data loading
dataloaders = {
    'train': data.DataLoader(dataset['train'], batch_size=bs, shuffle=True,
                             num_workers=0, pin_memory=True, drop_last=True),
    'valid': data.DataLoader(dataset['valid'], batch_size=bs, shuffle=True,
                             num_workers=0, pin_memory=True, drop_last=True),
    'test': data.DataLoader(dataset['test'], batch_size=bs, shuffle=True,
                            num_workers=0, pin_memory=True, drop_last=True)
}

# Print the train and validation data sizes
print("Training-set size:", dataset_sizes['train'],
      "\nValidation-set size:", dataset_sizes['valid'],
      "\nTesting-set size:", dataset_sizes['test'])

# Set default device as gpu, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 這裡新增model種類
#########################################################################
option = int(input("\nWhich model want to apply: 1 for Resnet :"))

if option == 1:
    model_ft = models.resnet18(pretrained=True)

else:
    raise ValueError("Invalid option!")

#########################################################################

num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)


# Transfer the model to GPU
model_ft = model_ft.to(device)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

# optimizer = optim.Adam(model_ft.parameters(), lr=0.001)
# (optional: choosse Adam)
#                                               -- 蕭望緯 --

# Learning rate decay
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)


# Train the model
model_ft = model_method.train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs, dataloaders, dataset_sizes, device)

model_method.test_model(model_ft, criterion, dataloaders, dataset_sizes, device)

################################### END ###################################
