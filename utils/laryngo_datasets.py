import json
import os
from pathlib import Path

import numpy as np
from torchvision import datasets, transforms

from .datasets import AllowEmptyImageFolder, split_images_labels
from .transforms import StrongTrivialAugment, StrongRotateTrivialAugment


def _data(*parts: str) -> str:
    """Resolve a dataset path under CAILOOP_DATA_ROOT.

    Clinical images are not included in this repository.
    """
    root = os.environ.get("CAILOOP_DATA_ROOT")
    if not root:
        raise FileNotFoundError(
            "Set CAILOOP_DATA_ROOT to the directory that contains the image "
            "folders (v5/, prospective/, cil/, loc/, diag/). Clinical images "
            "are not included in this repository."
        )
    return str(Path(root).joinpath(*parts))


def _get_idata(dataset_name):
    name = dataset_name.lower()
    if name == "laryngo":
        return iLaryngo29()
    elif name == "laryngo_holdout":
        return iLaryngo_holdout()
    elif name == "laryngo_val_new":
        return iLaryngo_val_new()
    elif name == "laryngo_multi":
        return iLaryngo3()
    elif name == "laryngo_loc":
        return iLaryngo_loc()
    elif name == "laryngo_cil":
        return iLaryngo_CIL()
    elif name == "laryngo_diag":
        return iLaryngo_diag_39()
    elif name == "laryngo_prospective":
        return iLaryngo_prospective()
    elif name == "laryngo_stages":
        return iLaryngo_stages()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


class iLaryngo29(object):
    use_path = True
    train_trsf = [
        transforms.Resize([224, 224]),
        transforms.TrivialAugmentWide(),
    ]
    test_trsf = [
        transforms.Resize([224, 224]),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(30).tolist()

    def download_data(self):
        train_dir = _data("v5", "train")
        mem_dir = train_dir
        test_dir = _data("v5", "test")

        train_dset = AllowEmptyImageFolder(train_dir)
        mem_dset = AllowEmptyImageFolder(mem_dir)
        test_dset = AllowEmptyImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.mem_data, self.mem_targets = split_images_labels(mem_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class _iLaryngo30ImageFolderSplit(object):
    """Shared 30-class ImageFolder loader; subclasses set train/test roots."""

    use_path = True
    train_trsf = [
        transforms.Resize([224, 224]),
        transforms.TrivialAugmentWide(),
    ]
    test_trsf = [
        transforms.Resize([224, 224]),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    class_order = np.arange(30).tolist()
    train_dir = None
    test_dir = None

    def download_data(self):
        train_dir = _data(*self.train_dir)
        test_dir = _data(*self.test_dir)
        mem_dir = train_dir

        train_dset = AllowEmptyImageFolder(train_dir)
        mem_dset = AllowEmptyImageFolder(mem_dir)
        test_dset = AllowEmptyImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.mem_data, self.mem_targets = split_images_labels(mem_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iLaryngo_holdout(_iLaryngo30ImageFolderSplit):
    """Final Internal reporting split (75% of remapped test_new)."""

    train_dir = ("v5", "train_new")
    test_dir = ("v5", "test_holdout")


class iLaryngo_val_new(_iLaryngo30ImageFolderSplit):
    """Internal tuning split (25% of remapped test_new); not for final fig2 Internal."""

    train_dir = ("v5", "train_new")
    test_dir = ("v5", "val_new")


class iLaryngo_prospective(object):
    use_path = True
    train_trsf = [
        transforms.Resize([224, 224]),
        transforms.TrivialAugmentWide(),
    ]
    test_trsf = [
        transforms.Resize([224, 224]),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(30).tolist()

    def download_data(self):
        train_dir = _data("prospective", "dataset")
        mem_dir = train_dir
        test_dir = train_dir

        train_dset = AllowEmptyImageFolder(train_dir)
        mem_dset = AllowEmptyImageFolder(mem_dir)
        test_dset = AllowEmptyImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.mem_data, self.mem_targets = split_images_labels(mem_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iLaryngo3(object):
    use_path = True
    train_trsf = [
        transforms.Resize([224, 224]),
        transforms.TrivialAugmentWide(),
    ]
    test_trsf = [
        transforms.Resize([224, 224]),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(3).tolist()

    def download_data(self):
        train_dir = _data("multi", "weihai")
        mem_dir = train_dir
        test_dir = train_dir

        train_dset = datasets.ImageFolder(train_dir)
        mem_dset = datasets.ImageFolder(mem_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.mem_data, self.mem_targets = split_images_labels(mem_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iLaryngo_CIL(object):
    use_path = True
    train_trsf = [
        transforms.Resize([224, 224]),
        StrongTrivialAugment(),
    ]
    test_trsf = [
        transforms.Resize([224, 224]),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    class_order = np.arange(6).tolist()

    def download_data(self):
        train_dir = _data("cil", "train")
        mem_dir = train_dir
        test_dir = _data("cil", "test")

        train_dset = datasets.ImageFolder(train_dir)
        mem_dset = datasets.ImageFolder(mem_dir)
        test_dset = datasets.ImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.mem_data, self.mem_targets = split_images_labels(mem_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iLaryngo_loc(object):
    use_path = True
    train_trsf = [
        transforms.Resize([224, 224]),
        StrongTrivialAugment(),
    ]
    test_trsf = [
        transforms.Resize([224, 224]),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    nclass = 17
    class_order = np.arange(nclass).tolist()

    def download_data(self):
        train_dir = _data("loc", "train")
        test_dir = _data("loc", "val")
        mem_dir = train_dir

        train_dset = AllowEmptyImageFolder(train_dir)
        mem_dset = AllowEmptyImageFolder(mem_dir)
        test_dset = AllowEmptyImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.mem_data, self.mem_targets = split_images_labels(mem_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

class iLaryngo_diag_39(object):
    use_path = True
    train_trsf = [
        transforms.Resize([224, 224]),
        # StrongRotateTrivialAugment(),
        StrongTrivialAugment(),
    ]
    test_trsf = [
        transforms.Resize([224, 224]),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    nclass = 48
    class_order = np.arange(nclass).tolist()

    def download_data(self):
        train_dir = _data("diag", "train")
        test_dir = _data("diag", "test")
        mem_dir = train_dir

        train_dset = AllowEmptyImageFolder(train_dir)
        mem_dset = AllowEmptyImageFolder(mem_dir)
        test_dset = AllowEmptyImageFolder(test_dir)

        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.mem_data, self.mem_targets = split_images_labels(mem_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iLaryngo_stages(object):
    """30-class train_new / test_new, train subset controlled by stages_v1.json."""

    use_path = True
    train_trsf = [
        transforms.Resize([224, 224]),
        StrongTrivialAugment(),
    ]
    test_trsf = [
        transforms.Resize([224, 224]),
    ]
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    nclass = 30
    class_order = np.arange(nclass).tolist()

    train_root = ("v5", "train_new")
    test_root = ("v5", "test_new")
    stages_manifest = ("v5", "train_new", "stages_v1.json")
    # Override before DataManager(...), e.g. iLaryngo_stages.train_stages = [1, 2, 3, 4, 5]
    train_stages = list(range(1, 6))

    @staticmethod
    def _class_to_index(class_name: str) -> int:
        return int(class_name.split("_", 1)[0])

    def _load_stage_split(self):
        train_root = Path(_data(*self.train_root))
        with open(_data(*self.stages_manifest), "r", encoding="utf-8") as f:
            manifest = json.load(f)

        case_map = {c["case_key"]: c for c in manifest["cases"]}
        stage_set = {int(s) for s in self.train_stages}

        images, labels = [], []
        for stage in sorted(stage_set):
            for case_key in manifest["by_stage"][str(stage)]:
                case = case_map[case_key]
                label = self._class_to_index(case["class"])
                for rel in case["images"]:
                    images.append(str(train_root / rel))
                    labels.append(label)

        return np.array(images), np.array(labels, dtype=np.int64)

    def download_data(self):
        self.train_data, self.train_targets = self._load_stage_split()
        self.mem_data, self.mem_targets = self.train_data, self.train_targets

        test_dset = AllowEmptyImageFolder(_data(*self.test_root))
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)
