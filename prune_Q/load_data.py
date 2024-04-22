import torch
import torchvision.transforms as transforms
import os
import torchvision.datasets as dset
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
import torch.nn as nn


def load_CIFAR10_dataset(batch_size: int = 128, calibration_batch_size: int = 1024,
                         data_path: str = './data'):
    """
    download and loading the data loaders
    Args:
        batch_size: batch size for train and test loader
        calibration_batch_size: size of the calibration batch
        data_path: directory to save data

    Returns:
        train_loader, test_loader, calibration_loader
    """
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    num_workers = os.cpu_count()

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_data = dset.CIFAR10(data_path,
                              train=True,
                              transform=train_transform,
                              download=True)
    test_data = dset.CIFAR10(data_path,
                             train=False,
                             transform=test_transform,
                             download=True)
    calibration_data = dset.CIFAR10(data_path,
                                    train=True,
                                    transform=test_transform,
                                    download=False)

    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    train_idx, calibration_idx = indices[calibration_batch_size:], indices[:calibration_batch_size]
    train_sampler = SubsetRandomSampler(train_idx)
    calibration_sampler = SubsetRandomSampler(calibration_idx)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True)
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)
    calibration_loader = DataLoader(
        calibration_data,
        batch_size=calibration_batch_size,
        sampler=calibration_sampler,
        num_workers=num_workers,
        pin_memory=True)

    return train_loader, test_loader, calibration_loader

def load_CIFAR10_dataset(batch_size: int = 128, calibration_batch_size: int = 1024,
                         data_path: str = './data'):
    """
    download and loading the data loaders
    Args:
        batch_size: batch size for train and test loader
        calibration_batch_size: size of the calibration batch
        data_path: directory to save data

    Returns:
        train_loader, test_loader, calibration_loader
    """
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    if not os.path.isdir(data_path):
        os.makedirs(data_path)
    num_workers = os.cpu_count()

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_data = dset.CIFAR10(data_path,
                              train=True,
                              transform=train_transform,
                              download=True)
    test_data = dset.CIFAR10(data_path,
                             train=False,
                             transform=test_transform,
                             download=True)
    calibration_data = dset.CIFAR10(data_path,
                                    train=True,
                                    transform=test_transform,
                                    download=False)

    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)

    train_idx, calibration_idx = indices[calibration_batch_size:], indices[:calibration_batch_size]
    train_sampler = SubsetRandomSampler(train_idx)
    calibration_sampler = SubsetRandomSampler(calibration_idx)

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True)
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True)
    calibration_loader = DataLoader(
        calibration_data,
        batch_size=calibration_batch_size,
        sampler=calibration_sampler,
        num_workers=num_workers,
        pin_memory=True)

    return train_loader, test_loader, calibration_loader

from functools import reduce


def get_module_by_name(module, access_string):
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)


def model_size(model):
    param_size = 0
    for name, p in model.named_parameters():
        if 'weight_orig' in name:
            m = get_module_by_name(model, name[:-12])
            if hasattr(m, 'weight_mask'):
                param_size_ = torch.count_nonzero(m.weight_mask) * p.element_size()
                param_size += param_size_
            else:
                param_size_ = p.nelement() * p.element_size()
                param_size += param_size_
        else:
            param_size_ = p.nelement() * p.element_size()
            param_size += param_size_

    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size) / 1024 ** 2
    return 'model size: {:.3f}MB'.format(size_all_mb)


@torch.inference_mode()
def evaluate(
        model: nn.Module,
        dataflow: DataLoader,
        device: torch.device = torch.device("cuda")
) -> float:
    model.eval()

    num_samples = 0
    num_correct = 0

    for inputs, targets in dataflow:
        # Move the data from CPU to GPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Inference
        outputs = model(inputs)

        # Convert logits to class indices
        outputs = outputs.argmax(dim=1)

        # Update metrics
        num_samples += targets.size(0)
        num_correct += (outputs == targets).sum()

    return (num_correct / num_samples * 100).item()