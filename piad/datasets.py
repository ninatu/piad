import os
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import skimage.io
import skimage.transform
import re
import tqdm


def inf_dataloader(data_loader):
    while True:
        for data in data_loader:
            yield data


class _GrayscaleDataset(Dataset):
    processed_folder = 'processed'
    train_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(self, root, train=True, download=False):
        super(_GrayscaleDataset, self).__init__()
        assert download == False

        if train:
            self.data, self.targets = torch.load(os.path.join(root, self.processed_folder, self.train_file))
        else:
            self.data, self.targets = torch.load(os.path.join(root, self.processed_folder, self.test_file))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(), mode='L')
        return img, target

    def __len__(self):
        return len(self.data)


class _ColorDataset(_GrayscaleDataset):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img.numpy(), mode='RGB')
        return img, target


class _NormalAnomalyBase(Dataset):
    def __init__(self, DatasetClass, root, split, normal_class, normal=True, transform=None, download=True,
                 equal_abnormal_count=False):
        super().__init__()

        if split == 'train':
            dataset = DatasetClass(root=root, train=True, download=download)
        else:
            dataset = DatasetClass(root=root, train=False, download=download)

        self.data = dataset
        self.transform = transform
        self.normal_class = normal_class

        if split == 'train':
            self._min_dataset_size = 10000  # to speed up training
        else:
            self._min_dataset_size = 0

        if normal:
            self.active_indexes = [i for i in range(len(dataset)) if dataset.targets[i] == self.normal_class]
        else:
            self.active_indexes = [i for i in range(len(dataset)) if dataset.targets[i] != self.normal_class]

            if equal_abnormal_count:
                normal_count = (dataset.targets == self.normal_class).sum()
                np.random.shuffle(self.active_indexes)
                self.active_indexes = self.active_indexes[:normal_count]

    def __getitem__(self, index):
        index = index % len(self.active_indexes)
        image, _ = self.data[self.active_indexes[index]]

        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        _len = len(self.active_indexes)
        if _len >= self._min_dataset_size:
            return _len
        else:
            # if data is too small we just duplicate all samples several times
            factor = int(np.ceil(self._min_dataset_size / _len))
            return factor * _len


class CIFAR10(_NormalAnomalyBase):
    def __init__(self, root, split, normal_class, normal=True, transform=None):
        super().__init__(datasets.CIFAR10, root, split, normal_class, normal, transform, download=True)


class MNIST(_NormalAnomalyBase):
    def __init__(self, root, split, normal_class, normal=True, transform=None):
        super().__init__(datasets.MNIST, root, split, normal_class, normal, transform, download=True)


class FashionMNIST(_NormalAnomalyBase):
    def __init__(self, root, split, normal_class, normal=True, transform=None):

        preprocessed_root = os.path.join(root, 'ad_protocol')
        fashion_mnist_preprocessed(root, preprocessed_root)

        super().__init__(_GrayscaleDataset, preprocessed_root, split, normal_class, normal, transform,
                         download=False,
                         equal_abnormal_count=True)


class COIL100(_NormalAnomalyBase):
    def __init__(self, root, split, normal_class, normal=True, transform=None):
        preprocessed_root = os.path.join(root, 'ad_protocol')
        coil_100_preprocessing(root, preprocessed_root)

        super().__init__(_ColorDataset, preprocessed_root, split, normal_class, normal, transform,
                         download=False,
                         equal_abnormal_count=True)


class CelebA(_NormalAnomalyBase):
    def __init__(self, root, split, normal_class, normal=True, transform=None, abnormal_class=None,
                 extended_attribute_list=False):
        assert normal_class == 0

        self.data = datasets.CelebA(root, split)
        self.transform = transform

        if extended_attribute_list:
            self.attributes = ["Bags_Under_Eyes", "Bald", "Bangs", "Eyeglasses", "Goatee",
                                                  "Heavy_Makeup", "Mustache", "Sideburns", "Wearing_Hat"]
        else:
            self.attributes = ["Bald", "Mustache", "Bangs", "Eyeglasses", "Wearing_Hat"]

        if normal:
            byte_index = torch.ones(len(self.data), dtype=torch.bool)
            for attr_name in self.attributes:
                byte_index = byte_index.logical_and(self.data.attr[:, self.data.attr_names.index(attr_name)] == 0)
            self.active_indexes = torch.nonzero(byte_index, as_tuple=False).numpy().flatten()
        else:
            assert abnormal_class in self.attributes
            # filter images where this attribute is presented
            byte_index = self.data.attr[:, self.data.attr_names.index(abnormal_class)] == 1
            # filter images where all other attributes are not presented
            for attr_name in self.attributes:
                if attr_name != abnormal_class:
                    byte_index = byte_index.logical_and(self.data.attr[:, self.data.attr_names.index(attr_name)] == 0)

            self.active_indexes = torch.nonzero(byte_index, as_tuple=False).numpy().flatten()

        if split == 'train':
            self._min_dataset_size = 10000  # as required in _NormalAnomalyBase
        else:
            self._min_dataset_size = 0


class LSUN(_NormalAnomalyBase):
    def __init__(self, root, split, normal_class, normal=True, transform=None):
        assert normal_class == 0

        self.transform = transform

        _class = 'bedroom' if normal else "conference_room"
        if split == 'test':
            split = 'val' # as was done in ADGAN (https://link.springer.com/chapter/10.1007/978-3-030-10925-7_1)

        self.data = datasets.LSUN(root, classes=[f"{_class}_{split}"])
        self.active_indexes = list(range(len(self.data)))  # as required in _NormalAnomalyBase

        if split == 'train':
            self._min_dataset_size = 10000  # as required in _NormalAnomalyBase
        else:
            self._min_dataset_size = 0


def fashion_mnist_preprocessed(original_root, preprocessed_root):
    train = datasets.FashionMNIST(root=original_root, train=True, download=True)
    test = datasets.FashionMNIST(root=original_root, train=False, download=True)

    if os.path.exists(preprocessed_root):
        print('Preprocessing to fit to the anomaly detection train/test protocol was already done.')
    else:
        print("Preprocessing to fit to the anomaly detection train/test protocol....")
        data = torch.cat((train.data, test.data), dim=0)
        targets = torch.cat((train.targets, test.targets), dim=0)

        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        indexs = np.array(range(data.shape[0]))
        for label in range(10):
            label_indexs = indexs[targets.numpy() == label]
            np.random.shuffle(label_indexs)
            n = int(len(label_indexs) * 0.8)
            train_indexs = label_indexs[:n]
            test_indexs = label_indexs[n:]

            train_data.append(data[train_indexs])
            train_labels.append(targets[train_indexs])

            test_data.append(data[test_indexs])
            test_labels.append(targets[test_indexs])

        train_data = torch.cat(train_data, dim=0).detach().clone()
        train_labels = torch.cat(train_labels, dim=0).detach().clone()
        test_data = torch.cat(test_data, dim=0).detach().clone()
        test_labels = torch.cat(test_labels, dim=0).detach().clone()

        train = (train_data, train_labels)
        test = (test_data, test_labels)

        output_root = os.path.join(preprocessed_root, 'processed')
        os.makedirs(output_root, exist_ok=True)
        torch.save(train, os.path.join(output_root, 'training.pt'))
        torch.save(test, os.path.join(output_root, 'test.pt'))
        print("Preprocessing is done.")


def coil_100_preprocessing(original_root, preprocessed_root):
    if os.path.exists(preprocessed_root):
        print('Preprocessing to fit to the anomaly detection train/test protocol was already done.')
    else:
        print("Preprocessing to fit to the anomaly detection train/test protocol....")

        original_root = os.path.join(original_root, 'coil-100')
        pat = 'obj(?P<class>\d+)__(?P<numb>\d+)'
        imgs = []
        labels = []

        print("Loading data....")
        for filename in tqdm.tqdm(os.listdir(original_root)):
            if os.path.splitext(filename)[1] != '.png':
                continue
            img = skimage.io.imread(os.path.join(original_root, filename))
            img = skimage.transform.rescale(img, (0.25, 0.25, 1))
            img = (img * 255).astype(np.uint8)
            label = int(re.match(pat, filename)['class'])
            if label == 100:
                label = 0
            imgs.append(img)
            labels.append(label)

        imgs = np.stack(imgs, axis=0)
        labels = np.stack(labels, axis=0)

        data = torch.from_numpy(imgs)
        labels = torch.from_numpy(labels)

        train_data = []
        train_labels = []
        test_data = []
        test_labels = []

        indexs = np.array(range(data.shape[0]))
        for label in np.unique(labels):
            label_indexs = indexs[labels.numpy() == label]
            np.random.shuffle(label_indexs)
            n = int(len(label_indexs) * 0.8)
            train_indexs = label_indexs[:n]
            test_indexs = label_indexs[n:]

            train_data.append(data[train_indexs])
            train_labels.append(labels[train_indexs])

            test_data.append(data[test_indexs])
            test_labels.append(labels[test_indexs])

        train_data = torch.cat(train_data, dim=0).detach().clone()
        train_labels = torch.cat(train_labels, dim=0).detach().clone()
        test_data = torch.cat(test_data, dim=0).detach().clone()
        test_labels = torch.cat(test_labels, dim=0).detach().clone()

        train = (train_data, train_labels)
        test = (test_data, test_labels)

        os.makedirs(os.path.join(preprocessed_root, 'processed'), exist_ok=True)

        torch.save(train, os.path.join(preprocessed_root, 'processed', 'training.pt'))
        torch.save(test, os.path.join(preprocessed_root, 'processed', 'test.pt'))
        print("Preprocessing is done.")


def create_dataset(dataset_type, dataset_root, split, normal_class, normal, transform=None, abnormal_class=None,
                   extra_dataset_params=None):
    if extra_dataset_params is None:
        extra_dataset_params = {}

    if dataset_type == 'mnist':
        dataset = MNIST(dataset_root, split, normal_class, normal, transform)
    elif dataset_type == 'fashion_mnist':
        dataset = FashionMNIST(dataset_root, split, normal_class, normal, transform)
    elif dataset_type == 'cifar10':
        dataset = CIFAR10(dataset_root, split, normal_class, normal, transform)
    elif dataset_type == 'coil100':
        dataset = COIL100(dataset_root, split, normal_class, normal, transform)
    elif dataset_type == 'celeba':
        dataset = CelebA(dataset_root, split, normal_class, normal, transform, abnormal_class=abnormal_class,
                         **extra_dataset_params)
    elif dataset_type == 'lsun':
        dataset = LSUN(dataset_root, split, normal_class, normal, transform)
    else:
        raise ValueError('Unknown data type: {}'.format(dataset_type))
    return dataset


def create_transform(dataset_type):
    if dataset_type == 'mnist' or dataset_type == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    elif dataset_type == 'cifar10' or dataset_type == 'coil100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif dataset_type == 'celeba':
        transform = transforms.Compose([
            transforms.CenterCrop((140, 140)),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    elif dataset_type == 'lsun':
        transform = transforms.Compose([
            transforms.CenterCrop((256, 256)),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        raise ValueError('Unknown data type: {}'.format(dataset_type))
    return transform
