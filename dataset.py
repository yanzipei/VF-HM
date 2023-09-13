import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image

eye_types = ['L', 'R']
vf_types = ['vector', 'array']

L_mask = np.asarray([[False, False, False, False, False, False, False, False, False, False],
                     [False, False, False, True, True, True, True, False, False, False],
                     [False, False, True, True, True, True, True, True, False, False],
                     [False, True, True, True, True, True, True, True, True, False],
                     [False, True, False, True, True, True, True, True, True, True],
                     [False, True, False, True, True, True, True, True, True, True],
                     [False, True, True, True, True, True, True, True, True, False],
                     [False, False, True, True, True, True, True, True, False, False],
                     [False, False, False, True, True, True, True, False, False, False],
                     [False, False, False, False, False, False, False, False, False, False]])

R_mask = np.flip(L_mask, axis=1)


def vf_array_to_vector(vf_arr, eye_type: str):
    mask = L_mask if eye_type == 'L' else R_mask
    return vf_arr[mask]


def vf_vector_to_array(vf_vec: np.ndarray, eye_type: str, fill_value=np.nan):
    assert eye_type in eye_types

    vf_array = np.empty([10, 10])
    vf_array.fill(fill_value)

    if eye_type == 'L':
        mask = L_mask
    elif eye_type == 'R':
        mask = R_mask
    else:
        raise NotImplementedError
    vf_array[mask] = vf_vec

    return vf_array


class VFDataset(torch.utils.data.Dataset):
    def __init__(self,
                 csv_file,
                 fundus_dir,
                 vf_dir,
                 vf_type: str,
                 fundus_transform=None,
                 vf_transform=None,
                 eye_type: str = 'L'):

        assert eye_type in eye_types
        assert vf_type in vf_types

        super(VFDataset, self).__init__()

        # load df
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        self.fundus_dir = fundus_dir
        self.vf_dir = vf_dir

        self.vf_type = vf_type
        self.fundus_transform = fundus_transform
        self.vf_transform = vf_transform
        self.eye_type = eye_type

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        data = self.df.iloc[idx]

        fundus_file = os.path.join(self.fundus_dir, data['fundus_id'])
        fundus = Image.open(fundus_file)  # PIL Image

        vf_file = os.path.join(self.fundus_dir, data['vf_id'])
        vf = json.load(vf_file)  # np.ndarray

        eye_type = data['eye_type']

        # consistent
        if eye_type != self.eye_type:
            # flip
            fundus = fundus.transpose(Image.FLIP_LEFT_RIGHT)

        if self.fundus_transform is not None:
            fundus = self.fundus_transform(fundus)

        # consistent
        if eye_type != self.eye_type:
            vf = np.flip(vf, axis=1).copy()

        if self.vf_type == 'vector':
            vf = vf_array_to_vector(vf, self.eye_type)

        if self.vf_transform is not None:
            # vf_vector = self.target_transform(vf_vector)
            vf = self.vf_transform(vf)

        # nan to 0.
        vf = np.nan_to_num(vf, 0.)

        return fundus, vf


class VFDatasetVFHM(torch.utils.data.Dataset):
    def __init__(self,
                 csv_file,
                 fundus_dir,
                 vf_dir,
                 vf_type: str,
                 include_mm: bool,
                 fundus_transform=None,
                 vf_transform=None,
                 eye_type: str = 'L'):

        assert eye_type in eye_types
        assert vf_type in vf_types

        super(VFDatasetVFHM, self).__init__()
        assert vf_type == 'array'

        # load df
        self.csv_file = csv_file
        self.df = pd.read_csv(csv_file)
        self.fundus_dir = fundus_dir
        self.vf_dir = vf_dir

        self.include_mm = include_mm
        self.vf_type = vf_type
        self.fundus_transform = fundus_transform
        self.vf_transform = vf_transform
        self.eye_type = eye_type

        self.mm_classes = ['C0', 'C1', 'C2', 'C3', 'C4']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        data = self.df.iloc[idx]

        fundus_file = os.path.join(self.fundus_dir, data['fundus_id'])
        fundus = Image.open(fundus_file)  # PIL Image

        vf_file = os.path.join(self.fundus_dir, data['vf_id'])
        vf = json.load(vf_file)  # np.ndarray

        if self.include_mm:
            mm = data['mm']
            # mm to rank
            mm = self.mm_classes.index(mm)  # one hot
            mm = [1] * mm + [0] * (self.num_aux_classes - 1 - mm)
            mm = np.asarray(mm)

        eye_type = data['eye_type']

        # consistent
        if eye_type != self.eye_type:
            # flip
            fundus = fundus.transpose(Image.FLIP_LEFT_RIGHT)

        if self.fundus_transform is not None:
            fundus = self.fundus_transform(fundus)

        # consistent
        if eye_type != self.eye_type:
            vf = np.flip(vf, axis=1).copy()

        if self.vf_type == 'vector':
            vf = vf_array_to_vector(vf, self.eye_type)

        if self.vf_transform is not None:
            # vf_vector = self.target_transform(vf_vector)
            vf = self.vf_transform(vf)

        # nan to 0.
        vf = np.nan_to_num(vf, 0.)

        # vf to rank
        vf = vf.astype(int)
        # to rank
        vf_rank = []
        for row in vf:
            row_rank = []
            for item in row:
                item = [1] * item + [0] * (self.num_classes - 1 - item)
                row_rank.append(item)
            vf_rank.append(row_rank)

        vf = np.asarray(vf_rank)  # [h, w, num_classes - 1]
        vf = np.transpose(vf, (2, 0, 1))  # [num_classes - 1, h, w]

        if self.include_mm:
            return fundus, mm, vf
        else:
            return fundus, vf
