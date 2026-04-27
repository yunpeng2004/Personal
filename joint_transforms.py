import random

from PIL import Image
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF


class JointTransform:
    def __init__(self, enable=True):
        self.enable = enable
        self.color_jitter = ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        )

    def __call__(self, image, mask):
        if not self.enable:
            return image, mask

        if random.random() < 0.7:
            image = self.color_jitter(image)

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() < 0.5:
            angle = random.uniform(-12.0, 12.0)
            translate = (
                int(round(image.size[0] * random.uniform(-0.04, 0.04))),
                int(round(image.size[1] * random.uniform(-0.04, 0.04))),
            )
            scale = random.uniform(0.95, 1.05)
            image = TF.affine(
                image,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=0.0,
                interpolation=Image.BILINEAR,
                fill=0,
            )
            mask = TF.affine(
                mask,
                angle=angle,
                translate=translate,
                scale=scale,
                shear=0.0,
                interpolation=Image.NEAREST,
                fill=0,
            )

        k = random.randint(0, 3)
        if k > 0:
            angle = 90 * k
            image = TF.rotate(image, angle, interpolation=Image.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=Image.NEAREST)

        return image, mask
