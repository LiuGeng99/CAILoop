import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import default_loader, split_images_labels
from .laryngo_datasets import _get_idata


class DummyDataset(Dataset):
    def __init__(self, images, labels, trsf, use_path=False, aug=1):
        assert len(images) == len(labels), "Data size error!"
        if use_path:
            self.images = images
        else:
            print("loading dataset")
            images_list = []
            for img_path in images:
                images_list.append(np.array(Image.open(img_path).convert("RGB")))
            self.images = images_list

        self.aug = aug
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path

    def _get_nsamples_list(self):
        nsamples_dict = {}
        for label in self.labels:
            int_label = int(label)
            if int_label not in nsamples_dict:
                nsamples_dict[int_label] = 1
            else:
                nsamples_dict[int_label] += 1

        nsamples_list = []
        for cls in range(len(nsamples_dict)):
            nsamples_list.append(nsamples_dict[cls])

        return nsamples_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx] if self.use_path else None
        if self.aug == 1:
            if self.use_path:
                image = self.trsf(default_loader(self.images[idx]))
            else:
                image = self.trsf(Image.fromarray(self.images[idx]))
            label = self.labels[idx]
            return idx, image, label, img_path
        else:
            if self.use_path:
                images = [self.trsf(default_loader(self.images[idx])) for _ in range(self.aug)]
            else:
                images = [self.trsf(Image.fromarray(self.images[idx])) for _ in range(self.aug)]
            label = self.labels[idx]
            return idx, *images, label, img_path


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, aug=1):
        self.dataset_name = dataset_name
        self.aug = aug
        self._setup_data(dataset_name, shuffle, seed)
        print(self._class_order)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    def get_accumulate_tasksize(self, task):
        return sum(self._increments[: task + 1])

    def get_total_classnum(self):
        return len(self._class_order)

    def get_dataset(
        self, indices, source, mode, appendent=None, ret_data=False, m_rate=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "mem":
            x, y = self._mem_data, self._mem_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx + 1)
            else:
                class_data, class_targets = self._select_rmm(
                    x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate
                )
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(
                data, targets, trsf, self.use_path, self.aug if mode == "train" else 1
            )
        else:
            return DummyDataset(
                data, targets, trsf, self.use_path, self.aug if mode == "train" else 1
            )

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name)
        idata.download_data()

        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._mem_data, self._mem_targets = idata.mem_data, idata.mem_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order

        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._mem_targets = _map_new_class_index(self._mem_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]

        if isinstance(x, np.ndarray):
            x_return = x[idxes]
        else:
            x_return = []
            for id in idxes:
                x_return.append(x[id])
        return x_return, y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))
