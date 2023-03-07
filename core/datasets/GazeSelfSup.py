import re
import pandas as pd
import torch
import torchvision.transforms as transforms
import pickle
from typing import Any, Callable, List, Optional, Tuple
from pathlib import Path
import numpy as np
from PIL import Image
import json

from tqdm import tqdm

__all__ = ["GazeSelfSupDataset"]


def parse_sent(sent):
    res = re.sub("[^a-zA-Z-]", " ", sent)
    res = res.strip().lower().split()
    return res


class GazeSelfSupDataset(torch.utils.data.Dataset):
    def __init__(self, metadata, is_train=True):
        self.metadata = metadata
        self.is_train = is_train
        with open(self.metadata, "r") as f:
            self.data = json.load(f)
        self.dicom_ids = list(self.data.keys())
        # self.dicom_ids = self.dicom_ids[:5]

        self.max_fix_len = 400  # reflacx largest is 386, min is 15

    def __getitem__(self, index):
        """_summary_

        With dataloader, the shape will have the first dim = batchsize
        Args:
            index (_type_): _description_

        Returns:
            fixation: [400, 3]
            fix_mask: [400, 1]
        """        
        dicom_id = self.dicom_ids[index]
        fixation_path = self.data[dicom_id]["fixation_reflacx"]
        is_reflacx = True
        if self.__is_reflacx_empty(fixation_path):
            fixation_path = self.data[dicom_id]["fixation_egd"]
            is_reflacx = False
        else:
            # we only care about the first fixation per patient for now
            fixation_path = fixation_path[0]
        fixation_data = self.__read(fixation_path)
        fixation, ftimestamp_s, ftimestamp_e = self.__get_fixation_and_timestamp(
            fixation_data, is_reflacx
        )
        # fixation [fix_len, 2], duration [fix_len,1], torch cat them together to [fix_len, 3]
        fixation = torch.cat((fixation, ftimestamp_e - ftimestamp_s), dim=1)
        # norm time
        fixation[:, 2] = fixation[:, 2] / fixation[:, 2].max()
        fixation, fix_masks = self.__padding_mask(fixation)
        return fixation, fix_masks

    def __padding_mask(self, fixation):
        lf = len(fixation)
        if lf < self.max_fix_len:
            padding = torch.zeros(self.max_fix_len - lf, 3)
            fixation = torch.cat((fixation, padding), dim=0)
            fix_masks = torch.cat(
                (torch.ones(lf, 1), torch.zeros(self.max_fix_len - lf, 1)), dim=0
            )
        else:
            fixation = fixation[: self.max_fix_len]
            fix_masks = torch.ones(self.max_fix_len, 1)
        return fixation, fix_masks

    def __len__(self):
        return len(self.dicom_ids)

    def __get_fixation_and_timestamp(self, fixation_data, is_reflacx):
        """This function will get the fixation and timestamp from the fixation_data dictionary

        Args:
            fixation_data (_type_): _description_
            is_reflacx (bool): _description_
        """
        x = torch.tensor(fixation_data["x"])
        y = torch.tensor(fixation_data["y"])
        fixation = torch.stack((x, y), dim=1)
        # fixation = fixation / fixation.max()
        start_time = torch.tensor(fixation_data["start_time"]).unsqueeze(1)
        end_time = torch.tensor(fixation_data["end_time"]).unsqueeze(1)
        return fixation, start_time, end_time

    def __read(self, path):
        # get extension from path
        ext = Path(path).suffix
        if ext in [".csv"]:
            return pd.read_csv(path)
        elif ext in [".json"]:
            with open(path, "r") as f:
                return json.load(f)

    def __is_reflacx_empty(self, path_list: List[str]):
        return path_list[0] == "" and len(path_list) == 1


if __name__ == "__main__":
    ds = GazeSelfSupDataset(
        metadata="/home/ptthang/gaze_sample/data_here/reflacx_new_metadata.json",    )
    print(len(ds))
    from torch.utils.data import DataLoader

    dl = DataLoader(ds, batch_size=1, shuffle=True)
    # enumerate tqdm for the dataset
    mx, mn = 0, 100000
    for i, (fixation, fix_masks) in enumerate(tqdm(dl)):
        if i == 0:
            print(
                fixation.shape,
                fix_masks.shape,
            )
        if mx < fixation.shape[1]:
            mx = fixation.shape[1]
        if mn > fixation.shape[1]:
            mn = fixation.shape[1]
        # continue
        # print(img.shape, fixation.shape, transcript)
        break # only break for one sample, if it is ok, cmt break and run the whole dataset to make sure it runs

    print(mx, mn)
