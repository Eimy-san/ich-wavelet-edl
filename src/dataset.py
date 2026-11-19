"""PyTorch Dataset for RSNA-ICH brain-windowed CT slices."""
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from data_prep import load_rescale_map


class ICHDataset(Dataset):
    def __init__(self, split_csv, rescale_csv, img_size=224, wl=40, ww=80,
                 augment=False, channels=1):
        self.df = pd.read_csv(split_csv)
        self.img_size = img_size
        self.wl = wl
        self.ww = ww
        self.augment = augment
        self.channels = channels
        self.rescale = load_rescale_map(rescale_csv)

    def __len__(self):
        return len(self.df)

    def _parse_ids(self, path):
        parts = path.replace('\\', '/').split('/')
        pid = [p for p in parts if p.startswith('ID_')]
        if len(pid) >= 3:
            return pid[0], pid[1], pid[2]
        return None, None, None

    def _load_and_window(self, path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            img = np.zeros((512, 512), dtype=np.float32)
            return img
        img = img.astype(np.float32)
        pid, sid, rid = self._parse_ids(path)
        if (pid, sid, rid) in self.rescale:
            slope, intercept = self.rescale[(pid, sid, rid)]
        else:
            slope, intercept = 1.0, -32768.0
        hu = img * slope + intercept
        hu = np.clip(hu, self.wl - self.ww / 2, self.wl + self.ww / 2)
        hu = (hu - (self.wl - self.ww / 2)) / self.ww
        return hu

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self._load_and_window(row['image_path'])
        if self.augment:
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy()
            if np.random.rand() < 0.5:
                img = np.flipud(img).copy()
            angle = np.random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1.0)
            img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
            if np.random.rand() < 0.3:
                noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
                img = np.clip(img + noise, 0, 1)
        img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        if self.channels == 1:
            img = img[np.newaxis, :, :]
        else:
            img = np.stack([img] * 3, axis=0)
        img = torch.from_numpy(img.astype(np.float32))
        label = int(row['any'])
        subtypes = torch.tensor([
            int(row['epidural']), int(row['intraparenchymal']),
            int(row['intraventricular']), int(row['subarachnoid']),
            int(row['subdural'])
        ], dtype=torch.float32)
        return img, label, subtypes
