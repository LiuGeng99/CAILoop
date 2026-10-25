import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == "accimage":
        return accimage_loader(path)
    else:
        return pil_loader(path)


class AllowEmptyImageFolder(ImageFolder):
    """
    A modified ImageFolder that allows empty class directories.
    Maintains strict Folder -> Label ID mapping regardless of empty folders.
    """

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        """
        Finds the class folders in a dataset.
        Keeps all subdirectories as classes even if they contain no files.
        """
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    @staticmethod
    def make_dataset(
        directory: str,
        class_to_idx: Dict[str, int],
        extensions: Optional[Tuple[str, ...]] = None,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> List[Tuple[str, int]]:
        """
        Generates a list of samples of a form (path_to_sample, class_index).
        Gracefully skips empty folders without throwing exceptions per class.
        """
        directory = os.path.expanduser(directory)

        if is_valid_file is None:
            if extensions is not None:
                exts = tuple(ext.lower() for ext in extensions)
                def is_valid_file(x: str) -> bool:
                    return x.lower().endswith(exts)
            else:
                is_valid_file = lambda x: True

        instances = []

        for target_class, class_index in sorted(class_to_idx.items()):
            target_dir = os.path.join(directory, target_class)

            if not os.path.isdir(target_dir):
                continue

            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        instances.append((path, class_index))

        if not instances:
            raise RuntimeError(f"Found 0 valid files in all subfolders of {directory}.")

        return instances


def split_images_labels(imgs):
    images = []
    labels = []
    for item in imgs:
        images.append(item[0])
        labels.append(item[1])

    return np.array(images), np.array(labels)
