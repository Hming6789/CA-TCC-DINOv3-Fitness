import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .augmentations import DataTransform
from .video import VideoDataset


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = torch.as_tensor(dataset["samples"])
        y_train = torch.as_tensor(dataset["labels"]).long()

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        # MODIFIED: Nhận diện kênh theo cấu hình, không nhầm batch nhỏ với số kênh.
        if X_train.ndim != 3:
            raise ValueError("Original backbone requires (N,C,L) time series")
        if X_train.shape[1] != config.input_channels and X_train.shape[2] == config.input_channels:
            X_train = X_train.permute(0, 2, 1)
        if X_train.shape[1] != config.input_channels:
            raise ValueError("Time-series channel count does not match input_channels")

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]
        if training_mode == "self_supervised" or training_mode == "SupCon":  # no need to apply Augmentations in other modes
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if self.training_mode == "self_supervised" or self.training_mode == "SupCon":
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len


def data_generator(data_path, configs, training_mode):
    batch_size = configs.batch_size
    is_video = getattr(configs, "backbone_type", "original") == "dinov3"
    train_name = "pseudo_train_data" if training_mode == "SupCon" else "train"
    for percentage in (1, 5, 10, 50, 75):
        if f"_{percentage}p" in training_mode:
            train_name = f"train_{percentage}perc"
            break

    # MODIFIED: Video hỗ trợ manifest JSON hoặc tensor .pt; chuỗi gốc vẫn dùng .pt.
    def load_split(name, mode):
        manifest = os.path.join(data_path, name + ".json")
        if is_video and os.path.isfile(manifest):
            return VideoDataset(manifest, configs, mode)
        source = torch.load(os.path.join(data_path, name + ".pt"),
                            map_location="cpu", weights_only=True)
        dataset_class = VideoDataset if is_video else Load_Dataset
        return dataset_class(source, configs, mode)

    train_dataset = load_split(train_name, training_mode)
    valid_dataset = load_split("val", "supervised")
    test_dataset = load_split("test", "supervised")
    if not len(train_dataset) or batch_size < 1:
        raise ValueError("Training data and batch_size must be nonempty/positive")
    batch_size = min(batch_size, len(train_dataset))
    contrastive = training_mode in ("self_supervised", "SupCon", "dinov3_finetune")
    if contrastive and batch_size < 2:
        raise ValueError("Contrastive learning requires at least 2 clips per batch")
    workers = getattr(configs, "num_workers", 0)
    # Batch cuối một phần tử không hợp lệ với BatchNorm của projection head TC.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=contrastive or configs.drop_last, num_workers=workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              drop_last=False, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             drop_last=False, num_workers=workers)
    return train_loader, valid_loader, test_loader
