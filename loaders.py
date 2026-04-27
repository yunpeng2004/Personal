import os
import random

import torch
from torch.utils.data import DataLoader

from dataset import PolypDataset
from joint_transforms import JointTransform


def load_datasets(img_size=352, seed=42, use_train_augmentation=True):
    base_path = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(base_path, "images")
    masks_dir = os.path.join(base_path, "masks")

    all_samples = PolypDataset.build_pairs(images_dir, masks_dir)

    rng = random.Random(seed)
    rng.shuffle(all_samples)

    total_size = len(all_samples)
    train_size = int(0.8 * total_size)
    remain = total_size - train_size
    val_size = remain // 2
    test_size = remain - val_size

    train_samples = all_samples[:train_size]
    val_samples = all_samples[train_size:train_size + val_size]
    test_samples = all_samples[train_size + val_size:]

    train_dataset = PolypDataset(
        train_samples,
        image_size=img_size,
        joint_transform=JointTransform(enable=use_train_augmentation),
    )
    val_dataset = PolypDataset(
        val_samples,
        image_size=img_size,
        joint_transform=None,
    )
    test_dataset = PolypDataset(
        test_samples,
        image_size=img_size,
        joint_transform=None,
    )

    print(
        f"Dataset split | total={total_size} train={len(train_samples)} "
        f"val={len(val_samples)} test={len(test_samples)}"
    )

    return train_dataset, val_dataset, test_dataset


def create_dataloaders(batch_size, train_dataset, val_dataset, test_dataset, num_workers=0):
    use_cuda = torch.cuda.is_available()
    common_kwargs = {
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }
    if num_workers > 0:
        common_kwargs["persistent_workers"] = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )
    return train_loader, val_loader, test_loader
