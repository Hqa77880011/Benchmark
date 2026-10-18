import os
from pathlib import Path
from PIL import Image
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator(object):
    def __init__(self, output_size, training=True):
        self.output_size = output_size
        self.training = training

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        if self.training and random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif self.training and random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y, _ = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            image = zoom(image, (self.output_size[0] / x,
                                 self.output_size[1] / y, 1), order=3)
            label = zoom(label, (self.output_size[0] / x,
                                 self.output_size[1] / y), order=0)

        image = torch.from_numpy(image.astype(np.float32)).permute(2, 0, 1)  # C,H,W
        label = torch.from_numpy(label.astype(np.float32)).long()
        return {'image': image, 'label': label}


class Synapse_dataset(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None):
        self.transform = transform
        self.split = split
        self.data_dir = Path(base_dir)
        self.png_root = self.data_dir / split
        if (self.png_root / "img").is_dir():
            self.sample_list = sorted(p.name for p in (self.png_root / "img").glob("*.png"))
        else:
            if list_dir is None:
                raise ValueError("NPZ input requires --list_dir; PNG input needs <root>/<split>/img.")
            with open(Path(list_dir) / f"{split}.txt", encoding="utf-8") as stream:
                self.sample_list = [line.strip() for line in stream if line.strip()]
        if not self.sample_list:
            raise ValueError(f"Empty {split} dataset: {self.data_dir}")

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        slice_name = self.sample_list[idx].strip('\n').split(",")[0]
        if slice_name.endswith(".png"):
            with Image.open(self.png_root / "img" / slice_name) as source:
                image = np.array(source.convert("RGB"))
            with Image.open(self.png_root / "mask" / slice_name) as source:
                label = np.array(source)
            sample = {"image": image, "label": label}
            if self.transform:
                sample = self.transform(sample)
            sample["case_name"] = Path(slice_name).stem
            return sample
        if slice_name.endswith(".npz"):
            data_path = os.path.join(self.data_dir, slice_name)
        else:
            data_path = os.path.join(self.data_dir, slice_name + '.npz')

        data = np.load(data_path)
        if 'image' in data and 'label' in data:
            image, label = data['image'], data['label']
        else:
            image, label = data['data'], data['seg']

        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = slice_name
        return sample
