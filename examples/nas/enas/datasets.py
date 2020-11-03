# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageFolder

def get_custom_dataset(train_dir, valid_dir):
    """ Load custom classification dataset using ImageFolder.
        The train and test directory should have sub directories with name equals to label names.

    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((32, 32))
    ])
    train_dataset = ImageFolder(root=train_dir, transform=transform)
    valid_dataset = ImageFolder(root=valid_dir, transform=transform)
    return train_dataset, valid_dataset
    

def get_dataset(cls, train_dir=None, valid_data=None):
    MEAN = [0.49139968, 0.48215827, 0.44653124]
    STD = [0.24703233, 0.24348505, 0.26158768]
    transf = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip()
    ]
    normalize = [
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ]

    train_transform = transforms.Compose(transf + normalize)
    valid_transform = transforms.Compose(normalize)

    if cls == "cifar10":
        dataset_train = CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        dataset_valid = CIFAR10(root="./data", train=False, download=True, transform=valid_transform)
    elif cls == "custom_classification":
        dataset_train, dataset_valid = get_custom_dataset(train_dir, valid_data)
    else:
        raise NotImplementedError
    return dataset_train, dataset_valid
