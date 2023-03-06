import torch
import torchvision.transforms as transforms
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image
__all__ = ['CIFAR10Dataset']


class CIFAR10Dataset():
    def __init__(self, datapath, metapath, is_train=True):
        """_summary_

        Args:
            datapath (str): _description_
            metapath (str): _description_
            is_train (bool, optional): _description_. Defaults to True.
        """
        self.transform = transforms.Compose(
            [transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.datalist = datapath
        self.meta = metapath
        self.is_train = is_train
        self.data: Any = []
        self.targets = []
        for file_name in self.datalist:
            with open(file_name, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        with open(self.meta, "rb") as infile:
            data = pickle.load(infile, encoding="latin1")
            self.classes = data['label_names']
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
        # print(self.targets, self.classes)
    



    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        # if self.target_transform is not None:
        #     target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    ds = CIFAR10Dataset(datapath = ['/media/aioz-thang/data3/aioz-thang/jvn/torchism/data/CIFAR10/cifar-10-batches-py/test_batch'], metapath='/media/aioz-thang/data3/aioz-thang/jvn/torchism/data/CIFAR10/cifar-10-batches-py/batches.meta')
    print(len(ds))
    # better use dataloader to test, because custom dataset may cause "out of range" for the index, but the dataloader will handle it. Besides, we always use dataloader to train the model
    from torch.utils.data import DataLoader
    dl = DataLoader(ds, batch_size=1, shuffle=True)
    for i, (im, lbl) in enumerate(dl):
        print(im.shape, lbl)
        break
