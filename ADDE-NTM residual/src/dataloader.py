# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import os, random
import numpy as np
from PIL import Image, ImageFile
import torch.utils.data as data
import torchvision.transforms as transforms
from src.utils import get_idx_label


class DataQuery(data.Dataset):
    """
    Load generated queries for evaluation. Each query consists of a reference image and an indicator vector
    The indicator vector consists of -1, 1 and 0, which means remove, add, not modify
    Args:
        file_root: path that stores preprocessed files (e.g. imgs_test.txt, see README.md for more explanation)
        img_root_path: path that stores raw images
        ref_ids: the file name of the generated txt file, which includes the indices of reference images
        query_inds: the file name of the generated txt file, which includes the indicator vector for queries.
        img_transform: transformation functions for img. Default: ToTensor()
        mode: the mode 'train' or 'test' decides to load training set or test set
    """
    def __init__(self, file_root,  img_root_path, ref_ids,  query_inds, img_transform=None,
                 mode='test'):
        super(DataQuery, self).__init__()

        self.file_root = file_root
        self.img_transform = img_transform
        self.img_root_path = img_root_path
        self.mode = mode
        self.ref_ids = ref_ids
        self.query_inds = query_inds

        if not self.img_transform:
            self.img_transform = transforms.ToTensor()

        self.img_data, self.label_data, self.ref_idxs, self.query_inds, self.attr_num = self._load_dataset()

    def _load_dataset(self):
        with open(os.path.join(self.file_root, "imgs_%s.txt" % self.mode)) as f:
            img_data = f.read().splitlines()

        label_data = np.loadtxt(os.path.join(self.file_root, "labels_%s.txt" % self.mode), dtype=int)

        query_inds = np.loadtxt(os.path.join(self.file_root, self.query_inds), dtype=int)
        # query_inds = query_inds.reshape(1, len(query_inds))
        ref_idxs = np.loadtxt(os.path.join(self.file_root, self.ref_ids), dtype=int)
        # ref_idxs = np.zeros(shape=(1,1), dtype=int)

        attr_num = np.loadtxt(os.path.join(self.file_root, "attr_num.txt"), dtype=int)

        assert len(img_data) == label_data.shape[0]

        return img_data, label_data, ref_idxs, query_inds, attr_num

    def __len__(self):
        return self.ref_idxs.shape[0]

    def __getitem__(self, index):

        ref_id = int(self.ref_idxs[index])
        img = Image.open(os.path.join(self.img_root_path, self.img_data[ref_id]))
        img = img.convert('RGB')

        if self.img_transform:
            img = self.img_transform(img)

        indicator = self.query_inds[index]

        return img, indicator


class Data(data.Dataset):
    """
    Load data for attribute predictor training (pre-training)
    Args:
        file_root: path that stores preprocessed files (e.g. imgs_train.txt, see README.md for more explanation)
        img_root_path: path that stores raw images
        img_transform: transformation functions for img. Default: ToTensor()
        mode: the mode 'train' or 'test' decides to load training set or test set
    """
    def __init__(self, file_root, img_root_path, img_transform=None, mode='train'):
        super(Data, self).__init__()

        self.file_root = file_root
        self.img_transform = img_transform
        self.img_root_path = img_root_path
        self.mode = mode

        if not self.img_transform:
            self.img_transform = transforms.ToTensor()

        self.img_data, self.label_data, self.attr_num = self._load_dataset()

    def _load_dataset(self):
        with open(os.path.join(self.file_root, "imgs_%s.txt" % self.mode)) as f:
            img_data = f.read().splitlines()

        label_data = np.loadtxt(os.path.join(self.file_root, "labels_%s.txt" % self.mode), dtype=int)
        assert len(img_data) == label_data.shape[0]

        attr_num = np.loadtxt(os.path.join(self.file_root, "attr_num.txt"), dtype=int)

        return img_data, label_data, attr_num

    def __len__(self):
        return self.label_data.shape[0]

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_root_path, self.img_data[index]))
        img = img.convert('RGB')
        if self.img_transform:
            img = self.img_transform(img)

        label_vector = self.label_data[index]  #one-hot

        return img, get_idx_label(label_vector, self.attr_num)


class DataTriplet(data.Dataset):
    """
    Load generated attribute manipulation triplets for training.
    Args:
        file_root: path that stores preprocessed files (e.g. imgs_train.txt, see README.md for more explanation)
        img_root_path: path that stores raw images
        triplet_name: the filename of generated txt file, which includes ids of sampled triplets
        mode: 'train' or 'valid'
        ratio: ratio to split train and validation set. Default: 0.9
    """
    def __init__(self, file_root, img_root_path, triplet_name, img_transform=None, mode='train', ratio=0.9):
        self.file_root = file_root
        self.img_transform = img_transform
        self.img_root_path = img_root_path
        self.mode = mode
        self.triplet_name = triplet_name
        self.ratio = ratio

        if self.img_transform is None:
            self.img_transform = transforms.ToTensor()

        self.triplets, self.triplets_inds, self.img_data, self.label_one_hot, self.attr_num = self._load_dataset()

    def _load_dataset(self):
        with open(os.path.join(self.file_root, "imgs_train.txt")) as f:
            img_data = f.read().splitlines()

        label_one_hot = np.loadtxt(os.path.join(self.file_root, "labels_train.txt"), dtype=int)
        assert len(img_data) == label_one_hot.shape[0]

        with open(os.path.join(self.file_root, "%s.txt" % self.triplet_name)) as f:
            triplets = f.read().splitlines()

        triplets_inds = np.loadtxt(os.path.join(self.file_root, "%s_ind.txt" % self.triplet_name), dtype=int)  #indicators

        N = int(len(triplets) * self.ratio) #split train/val

        attr_num = np.loadtxt(os.path.join(self.file_root, "attr_num.txt"), dtype=int)

        if self.mode == 'train':
            triplets_o = triplets[:N]
            triplets_inds_o = triplets_inds[:N]
        elif self.mode == 'valid':
            triplets_o = triplets[N:]
            triplets_inds_o = triplets_inds[N:]

        return triplets_o, triplets_inds_o, img_data,  label_one_hot, attr_num

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, index):

        ref_id, pos_id, neg_id = self.triplets[index].split(' ')
        idxs = {'ref': int(ref_id), 'pos': int(pos_id), 'neg': int(neg_id)}
        imgs = {}
        for key in idxs.keys():
            with Image.open(os.path.join(self.img_root_path, self.img_data[idxs[key]])) as img:
                img = img.convert('RGB')
                if self.img_transform:
                    img = self.img_transform(img)
                imgs[key] = img

        one_hots = {}
        for key in idxs.keys():
            one_hots[key] = self.label_one_hot[idxs[key]]

        indicator = self.triplets_inds[index]

        labels = {}
        for key in idxs.keys():
            labels[key] = get_idx_label(one_hots[key], self.attr_num)

        return imgs, \
               one_hots, \
               labels, \
               indicator


class DataQueryManip(data.Dataset):
    def __init__(self, file_root,  img_root_path, ref_ids,  query_inds, img_transform=None, num_image_query=1,
                 mode='test'):
        super(DataQueryManip, self).__init__()

        self.file_root = file_root
        self.img_transform = img_transform
        self.img_root_path = img_root_path
        self.mode = mode
        self.ref_ids = ref_ids
        self.query_inds = query_inds
        self.num_image_query = num_image_query
        self.idx_image_query = 0

        if not self.img_transform:
            self.img_transform = transforms.ToTensor()

        self.img_data, self.label_data, self.ref_idxs, self.query_inds, self.attr_num = self._load_dataset()

    def _load_dataset(self):
        with open(os.path.join(self.file_root, "imgs_%s.txt" % self.mode)) as f:
            img_data = f.read().splitlines()

        label_data = np.loadtxt(os.path.join(self.file_root, "labels_%s.txt" % self.mode), dtype=int)

        query_inds = np.loadtxt(os.path.join(self.file_root, self.query_inds), dtype=int)
        # query_inds = query_inds.reshape(1, len(query_inds))
        ref_idxs = np.loadtxt(os.path.join(self.file_root, self.ref_ids), dtype=int)
        # ref_idxs = np.zeros(shape=(1,1), dtype=int)

        attr_num = np.loadtxt(os.path.join(self.file_root, "attr_num.txt"), dtype=int)

        self.idx_query_images = random.sample([x for x in range(len(ref_idxs))], self.num_image_query)
        # self.query_images = [img_data[int(ref_idxs[i])] for i in self.idx_query_images]
        self.query_images = [img_data[int(ref_idxs[self.idx_query_images[i]])] for i in range(len(self.idx_query_images))]

        assert len(img_data) == label_data.shape[0]

        return img_data, label_data, ref_idxs, query_inds, attr_num

    def __len__(self):
        return self.ref_idxs.shape[0]

    def __getitem__(self, index):

        if index in self.list_index_query_indicator:
            ref_id = int(self.ref_idxs[self.idx_query_images[self.idx_image_query]])
            img = Image.open(os.path.join(self.img_root_path, self.img_data[ref_id]))
            img = img.convert('RGB')

            if self.img_transform:
                img = self.img_transform(img)

            indicator = self.query_inds[index]

            return True, img, indicator, self.img_data[ref_id]

        else:
            return False, [], self.query_inds[0], self.img_data[0]


class DataQueryManipMethod2(data.Dataset):
    def __init__(self, file_root,  img_root_path, ref_ids,  query_inds, img_transform=None, num_image_query=1,
                 mode='test'):
        super(DataQueryManipMethod2, self).__init__()

        self.file_root = file_root
        self.img_transform = img_transform
        self.img_root_path = img_root_path
        self.mode = mode
        self.ref_ids = ref_ids
        self.query_inds = query_inds
        self.num_image_query = num_image_query
        self.idx_image_query = 0

        if not self.img_transform:
            self.img_transform = transforms.ToTensor()

        self.img_data, self.label_data, self.ref_idxs, self.query_inds, self.attr_num = self._load_dataset()

    def _load_dataset(self):
        with open(os.path.join(self.file_root, "imgs_%s.txt" % self.mode)) as f:
            img_data = f.read().splitlines()

        label_data = np.loadtxt(os.path.join(self.file_root, "labels_%s.txt" % self.mode), dtype=int)

        query_inds = np.loadtxt(os.path.join(self.file_root, self.query_inds), dtype=int)
        # query_inds = query_inds.reshape(1, len(query_inds))
        ref_idxs = np.loadtxt(os.path.join(self.file_root, self.ref_ids), dtype=int)
        # ref_idxs = np.zeros(shape=(1,1), dtype=int)

        attr_num = np.loadtxt(os.path.join(self.file_root, "attr_num.txt"), dtype=int)

        self.idx_query_images = random.sample([x for x in range(len(ref_idxs))], self.num_image_query)
        # self.query_images = [img_data[int(ref_idxs[i])] for i in self.idx_query_images]
        self.query_images = [img_data[int(ref_idxs[self.idx_query_images[i]])] for i in range(len(self.idx_query_images))]

        self.idx_target_images = []
        for i in range(0, len(self.idx_query_images)):
            image_target_found = False
            while not image_target_found:
                index_image = random.randint(0, len(ref_idxs) - 1)
                if index_image != self.idx_query_images[i]:
                    self.idx_target_images.append(index_image)
                    image_target_found = True
        self.target_images = [img_data[int(ref_idxs[self.idx_target_images[i]])] for i in range(len(self.idx_target_images))]

        assert len(img_data) == label_data.shape[0]

        return img_data, label_data, ref_idxs, query_inds, attr_num

    def __len__(self):
        return self.ref_idxs.shape[0]

    def __getitem__(self, index):

        if index in self.list_index_query_indicator:
            ref_id = int(self.ref_idxs[self.idx_query_images[self.idx_image_query]])
            img = Image.open(os.path.join(self.img_root_path, self.img_data[ref_id]))
            img = img.convert('RGB')

            if self.img_transform:
                img = self.img_transform(img)

            indicator = self.query_inds[index]

            return True, img, indicator, self.img_data[ref_id]

        else:
            return False, [], self.query_inds[0], self.img_data[0]
