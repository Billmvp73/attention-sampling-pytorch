from collections import namedtuple
from functools import partial
import hashlib
import os
from PIL import Image
import torch
import urllib.request
from os import path
import sys
import zipfile
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import json

class STS:
    """The STS class reads the annotations and creates the corresponding
    Sign objects."""

    def __init__(self, directory, CLASSES_TO_IDX, annotations=None):
        
        # ensure_dataset_exists(directory)
        self.CLASSES_TO_IDX = CLASSES_TO_IDX
        self._directory = directory
        # self._inner = "Set{}".format(1 + ((seed + 1 + int(train)) % 2))
        # self._data = self._load_signs(self._directory, self._inner)
        self.annotations = annotations
        self._data = self._load_files()

    def _load_files(self):
            with open(self.annotations, "r") as f:
                annotations = json.load(f)
            images = []
            final_annotations = []
            for anno in annotations:
                name = anno["name"]
                img_labels = anno["labels"]
                scene_attr = anno["attributes"]
                if scene_attr["timeofday"] == "daytime":
                    target_anno = []
                    img_path = os.path.join(self._directory, name)
                    images.append(img_path)
                    for img_label in img_labels:
                        if img_label["category"] not in self.CLASSES_TO_IDX:
                            continue
                        attributes = img_label["attributes"]
                        if not attributes["occluded"] and not attributes["truncated"]:
                            box2d = img_label["box2d"]
                            target_anno.append([float(box2d["x1"]), float(box2d["y1"]), float(box2d["x2"]), float(box2d["y2"]), self.CLASSES_TO_IDX[img_label["category"]]])
                    if len(target_anno) == 0:
                        target_anno.append([-1, -1, -1, -1, 0])
                    final_annotations.append(target_anno)
            return list(zip(images, final_annotations))

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

def bbox_transform_inv(xmin, ymin, xmax, ymax):
    x_min = int(round(xmin))
    width = int(round(xmax)) - x_min
    y_min = int(round(ymin))
    height = int(round(ymax) - y_min)
    return [x_min, y_min, width, height]

class MultiBddDetection(Dataset):
    """Provide a Keras Sequence for the SpeedLimits dataset which is basically
    a filtered version of the STS dataset.
    Arguments
    ---------
        directory: str, The directory that the dataset already is or is going
                   to be downloaded in
        train: bool, Select the training or testing sets
        seed: int, The prng seed for the dataset
    """
    CLASSES_TO_IDX = {
        "empty": 0,
        "car": 1,
        "bus": 1,
        "truck": 1,
        "train": 1,
        "bike": 2,
        "person": 2,
        "rider": 2,
        "motor": 2,
        "traffic light": 3,
        "traffic sign": 3
    }
    CLASSES = [0, 1]

    def __init__(self, directory, split='train', scales = [1], scale_factor = 0.3, objects=None):
        cwd = os.getcwd().replace('dataset', '')
        directory = path.join(cwd, directory)
        self.directory = directory
        self.split = split
        self.anno_path = None
        self.objects = objects
        self.scales = scales
        self._scale_factor = scale_factor
        if self.split == "train" or self.split == "val":
            self.anno_path = os.path.join(self.directory, "bdd100k_labels_images_{}.json".format(split))
        self.directory = os.path.join(self.directory, split)
        self._data = self._filter(STS(self.directory, self.CLASSES_TO_IDX, self.anno_path), self.objects)
        if self.split=="train":
            self.image_transform = transforms.Compose([transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
                                                    #    transforms.RandomAffine(degrees=0,
                                                    #                            translate=(100 / 1280, 100 / 960)),
                                                       transforms.ToTensor()
                                                       # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                       ])
        else:
            self.image_transform = transforms.Compose([transforms.ToTensor()
                                                       # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                                       ])

        weights = make_weights_for_balanced_classes(self._data, len(self.CLASSES))
        weights = torch.DoubleTensor(weights)
        self.sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

    def _filter(self, data, objects):
        filtered = []
        for image, annos in data:
            if objects is None:
                categories = []
                for a in annos:
                    if a[-1] not in categories:
                        categories.append(a[-1])
                if self.CLASSES_TO_IDX["car"] not in categories:
                    categories = [0]
                if len(categories) == 1:
                # categories = [a[-1] for a in annos]
                    filtered.append((image, categories[0]))
            else:
                filtered.append((image, annos))
        return filtered

    # def _acceptable(self, signs):
    #     # Keep it as empty
    #     if not signs:
    #         return signs, True

    #     # Filter just the speed limits and sort them wrt visibility
    #     signs = sorted(s for s in signs if s.name in self.LIMITS)

    #     # No speed limit but many other signs
    #     if not signs:
    #         return None, False

    #     # Not visible sign so skip
    #     if signs[0].visibility != "VISIBLE":
    #         return None, False

    #     return signs, True

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        image, category = self._data[i]
        x_lows = []
        x_highs = []
        high_img = Image.open(image)
        high_img = self.image_transform(high_img)
        low_img = F.interpolate(high_img[None, ...], scale_factor=self._scale_factor, mode='bilinear')[0]
        for scale in self.scales:
            if scale == 1:
                x_high = high_img
                x_low = low_img
            else:
                x_high = F.interpolate(high_img[None, ...], scale_factor=scale, mode='bilinear')[0]
                x_low = F.interpolate(low_img[None, ...], scale_factor=scale, mode='bilinear')[0]
            x_lows.append(x_low)
            x_highs.append(x_high)
        # x_high = Image.open(image)
        # x_high = self.image_transform(x_high)

        # x_low = F.interpolate(x_high[None, ...], scale_factor=0.3, mode='bilinear')[0]
        return x_lows, x_highs, category

    @property
    def image_size(self):
        return self[0][0].shape[1:]

    @property
    def class_frequencies(self):
        """Compute and return the class specific frequencies."""
        freqs = np.zeros(len(self.CLASSES), dtype=np.float32)
        for image, category in self._data:
            freqs[category] += 1
        return freqs / len(self._data)

    def strided(self, N):
        """Extract N images almost in equal proportions from each category."""
        order = np.arange(len(self._data))
        np.random.shuffle(order)
        idxs = []
        cat = 0
        while len(idxs) < N:
            for i in order:
                image, category = self._data[i]
                if cat == category:
                    idxs.append(i)
                    cat = (cat + 1) % len(self.CLASSES)
                if len(idxs) >= N:
                    break
        return idxs


def make_weights_for_balanced_classes(images, num_classes):
    count = [0] * num_classes
    for item in images:
        count[item[1]] += 1
    weight_per_class = [0.] * num_classes
    N = float(sum(count))
    for i in range(num_classes):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[1]]
    return weight


def reverse_transform(inp):
    """ Do a reverse transformation. inp should be of shape [3, H, W] """
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    inp = (inp * 255).astype(np.uint8)

    return inp


if __name__ == '__main__':
    bdd_detection_dataset = MultiBddDetection('dataset/bdd_detection', split='val', scales=[1, 0.5, 0.25])

    bdd_detection_dataloader = DataLoader(bdd_detection_dataset, shuffle=False, batch_size=4)

    for i, (x_low, x_high, label) in enumerate(bdd_detection_dataloader):
        print(x_low)
