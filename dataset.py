from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def normalize_stem(stem: str) -> str:
    stem = stem.lower().strip()
    suffixes = [
        "_mask", "-mask", " mask",
        "_gt", "-gt", " gt",
        "_seg", "-seg",
        "_label", "-label",
        "_annotation", "-annotation",
    ]
    for suf in suffixes:
        if stem.endswith(suf):
            stem = stem[:-len(suf)]
            break
    return stem


class PolypDataset(Dataset):
    def __init__(self, samples, image_size=352, joint_transform=None):
        self.samples = samples
        self.image_size = image_size
        self.joint_transform = joint_transform

        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        image_resize_kwargs = {
            "size": (image_size, image_size),
            "interpolation": InterpolationMode.BILINEAR,
        }
        mask_resize_kwargs = {
            "size": (image_size, image_size),
            "interpolation": InterpolationMode.NEAREST,
        }

        try:
            image_resize = transforms.Resize(**image_resize_kwargs, antialias=True)
        except TypeError:
            image_resize = transforms.Resize(**image_resize_kwargs)

        mask_resize = transforms.Resize(**mask_resize_kwargs)

        self.image_transform = transforms.Compose([
            image_resize,
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.mask_transform = transforms.Compose([
            mask_resize,
            transforms.ToTensor(),
        ])

    @staticmethod
    def build_pairs(images_dir, masks_dir):
        images_dir = Path(images_dir)
        masks_dir = Path(masks_dir)

        image_files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
        mask_files = [p for p in masks_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]

        image_map = {normalize_stem(p.stem): p for p in image_files}
        mask_map = {normalize_stem(p.stem): p for p in mask_files}

        common_keys = sorted(set(image_map.keys()) & set(mask_map.keys()))
        if len(common_keys) == 0:
            raise RuntimeError("No matched image-mask pairs found in images/ and masks/.")

        return [(str(image_map[k]), str(mask_map[k])) for k in common_keys]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_path, mask_path = self.samples[idx]
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.joint_transform is not None:
            image, mask = self.joint_transform(image, mask)

        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()
        return image, mask
