import os
import numpy as np
from PIL import Image
from torch.utils import data
import torch
import torch.nn as nn


def get_labels(folders):
    labels = []
    for folder in folders:
        if ('normal' in folder.lower()):
            labels.append(0)
        else:
            labels.append(1)
    return labels


class Dataset_CRNN(data.Dataset):
    "Characterizes a dataset for PyTorch"

    def __init__(self, data_path, transform=None):
        "Initialization"
        self.data_path = data_path
        self.folders = os.listdir(data_path)
        self.labels = get_labels(self.folders)
        self.transform = transform

    def __len__(self):
        "Denotes the total number of samples"
        return len(self.folders)

    def read_images(self, path, selected_folder, use_transform):
        video_path = os.path.join(path, selected_folder)
        frames = os.listdir(video_path)
        frames.sort()
        X = []
        for frame in frames:
            image = Image.open(os.path.join(video_path, frame))
            if use_transform is not None:
                image = use_transform(image)

            X.append(image)
        X = torch.stack(X, dim=0)

        return X

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        folder = self.folders[index]

        # Load data
        # (input) spatial images
        X = self.read_images(self.data_path, folder, self.transform)
        # (labels) LongTensor are for int64 instead of FloatTensor
        y = torch.LongTensor([self.labels[index]])

        # print(X.shape)
        return X, y
