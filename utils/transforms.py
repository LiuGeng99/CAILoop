import random
import torchvision.transforms.functional as F


class StrongTrivialAugment:
    def __init__(self, num_bins=31):
        self.num_bins = num_bins

    def __call__(self, img):
        magnitude = random.randint(0, self.num_bins)
        m = magnitude / float(self.num_bins)
        sign = random.choice([-1, 1])

        ops = [
            lambda img: img,
            lambda img: F.autocontrast(img),
            lambda img: F.equalize(img),
            lambda img: F.invert(img),
            lambda img: F.posterize(img, bits=max(1, 8 - int(m * 4))),
            lambda img: F.solarize(img, threshold=255.0 - m * 255.0),

            lambda img: F.adjust_brightness(img, brightness_factor=1.0 + m * sign * 0.9),
            lambda img: F.adjust_contrast(img, contrast_factor=1.0 + m * sign * 0.9),
            lambda img: F.adjust_sharpness(img, sharpness_factor=1.0 + m * sign * 0.9),

            # lambda img: F.rotate(img, angle=m * 20.0 * sign),

            lambda img: F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[m * 16.0 * sign, 0.0]),
            lambda img: F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[0.0, m * 16.0 * sign]),

            lambda img: F.affine(img, angle=0.0, translate=[int(m * 30 * sign), 0], scale=1.0, shear=[0.0, 0.0]),
            lambda img: F.affine(img, angle=0.0, translate=[0, int(m * 30 * sign)], scale=1.0, shear=[0.0, 0.0]),
        ]

        op = random.choice(ops)
        return op(img)


import random
import torchvision.transforms.functional as F

class StrongRotateTrivialAugment:
    def __init__(self, num_bins=31):
        self.num_bins = num_bins

    def __call__(self, img):
        angle = random.choice([0.0, 90.0, 180.0, 270.0])
        if angle != 0.0:
            img = F.rotate(img, angle=angle, expand=True)

        magnitude = random.randint(0, self.num_bins)
        m = magnitude / float(self.num_bins)
        sign = random.choice([-1, 1])

        ops = [
            lambda image: image,
            lambda image: F.autocontrast(image),
            lambda image: F.equalize(image),
            lambda image: F.invert(image),
            lambda image: F.posterize(image, bits=max(1, 8 - int(m * 4))),
            lambda image: F.solarize(image, threshold=255.0 - m * 255.0),

            lambda image: F.adjust_brightness(image, brightness_factor=1.0 + m * sign * 0.9),
            lambda image: F.adjust_contrast(image, contrast_factor=1.0 + m * sign * 0.9),
            lambda image: F.adjust_sharpness(image, sharpness_factor=1.0 + m * sign * 0.9),

            lambda image: F.affine(image, angle=0.0, translate=[m * 16.0 * sign, 0.0], scale=1.0, shear=[0.0, 0.0]),
            lambda image: F.affine(image, angle=0.0, translate=[0.0, m * 16.0 * sign], scale=1.0, shear=[0.0, 0.0]),
        ]

        op = random.choice(ops)
        return op(img)