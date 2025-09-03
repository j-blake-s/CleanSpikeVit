import torch
import numpy as np
import math
from spikingjelly.datasets import cifar10_dvs

def split_to_train_test_set(train_ratio: float, origin_dataset: torch.utils.data.Dataset, num_classes: int, random_split: bool = False):
    label_idx = []
    for i in range(num_classes):
        label_idx.append([])

    for i, item in enumerate(origin_dataset):
      y = item[1]
      if isinstance(y, np.ndarray) or isinstance(y, torch.Tensor):
        y = y.item()
      label_idx[y].append(i)
    train_idx = []
    test_idx = []
    if random_split:
      for i in range(num_classes):
        np.random.shuffle(label_idx[i])

    for i in range(num_classes):
      pos = math.ceil(label_idx[i].__len__() * train_ratio)
      train_idx.extend(label_idx[i][0: pos])
      test_idx.extend(label_idx[i][pos: label_idx[i].__len__()])

    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)


def load_data(dataset_dir, batch_size, timesteps):
 
  origin_set = cifar10_dvs.CIFAR10DVS(root=dataset_dir, data_type='frame', frames_number=timesteps, split_by='number')
  dataset_train, dataset_test = split_to_train_test_set(0.9, origin_set, 10)

  train_loader = torch.utils.data.DataLoader(
    dataset=dataset_train,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    pin_memory=True)

  test_loader = torch.utils.data.DataLoader(
    dataset=dataset_test,
    batch_size=batch_size,
    shuffle=True,
    drop_last=False,
    pin_memory=True)

  return train_loader, test_loader