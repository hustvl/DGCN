import os
import os.path as osp
import sys
import numpy as np
import random
import collections
import torch
import torchvision
from scipy.ndimage import zoom
import cv2
from torch.utils import data
from cues_reader import CuesReader


class VOCDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255,cues_dir=None, cues_name=None):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip().split(' ') for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.cues_reader = CuesReader(cues_dir, cues_name)
        self.files = []
        for name,ids in self.img_ids:
            img_file = osp.join(self.root, "JPEGImages/%s" % name)
            mask = osp.join(self.root, "SegmentationClassAug/%s.png" % name.split('.')[0])
            self.files.append({
                "img": img_file,
                "id": ids,
                "mask" : mask
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label, cues):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        cues = cv2.resize(cues, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_NEAREST)
        return image, label, cues

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        mask = cv2.imread(datafiles["mask"], cv2.IMREAD_GRAYSCALE)
        image_id = datafiles["id"]
        labels, cues= self.cues_reader.get_forward(image_id)
        size = image.shape

        markers_new = np.ones((41, 41))*255
        pos = np.where(cues == 1)
        markers_new[pos[1], pos[2]] = pos[0]
        img_h, img_w, _ = image.shape
        markers = zoom(markers_new, (float(img_h) / markers_new.shape[0], float(img_w) / markers_new.shape[1]), order=0) 
        if self.scale:
            image, mask, markers = self.generate_scale_label(image, mask, markers)

        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w, _ = image.shape

        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(mask, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
            markers_pad = cv2.copyMakeBorder(markers, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(self.ignore_label,))
        else:
            img_pad, label_pad, markers_pad = image, mask, markers
        img_h, img_w = label_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        markers_pad = np.asarray(markers_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        markers_pad = zoom(markers_pad, (41.0 / markers_pad.shape[0], 41.0 / markers_pad.shape[1]), order=0) 
        cues_new = np.zeros(cues.shape)
        for class_i in range(cues.shape[0]):
            pos = np.where(markers_pad == class_i)
            if len(pos)==0:
                continue
            cues_new[class_i,pos[0],pos[1]] = 1
        image = image[:,:,[2, 1, 0]]
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            cues_new = cues_new[:, :, ::flip]
            label = label[:, ::flip]
        return image.copy(),cues_new.copy(), labels, label.copy()

class VOCRetrainDataSet(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=True, ignore_label=255):
        self.root = root
        self.list_path = list_path
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.is_mirror = mirror
        self.img_ids = [i_id.strip().split(' ') for i_id in open(list_path)]
        if not max_iters==None:
           self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        for image,label in self.img_ids:
            img_file = (self.root+image)
            label_file = label
            self.files.append({
                "img": img_file,
                "label": label_file,
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image):
        f_scale = 0.5 + random.randint(0, 11) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation = cv2.INTER_LINEAR)
        return image

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        if self.scale:
            image = self.generate_scale_label(image)

        image = np.asarray(image, np.float32)
        image -= self.mean
        img_h, img_w, _ = image.shape

        label = zoom(label, (float(img_h) / 41.0, float(img_w) / 41.0),order=0)
        pad_h = max(self.crop_h - img_h, 0)
        pad_w = max(self.crop_w - img_w, 0)
        if pad_h > 0 or pad_w > 0:
            img_pad = cv2.copyMakeBorder(image, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT, 
                value=(0.0, 0.0, 0.0))
            label_pad = cv2.copyMakeBorder(label, 0, pad_h, 0, 
                pad_w, cv2.BORDER_CONSTANT,
                value=(255.0))
        else:
            img_pad, label_pad = image, label

        img_h, img_w,_ = img_pad.shape
        h_off = random.randint(0, img_h - self.crop_h)
        w_off = random.randint(0, img_w - self.crop_w)
        image = np.asarray(img_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        label = np.asarray(label_pad[h_off : h_off+self.crop_h, w_off : w_off+self.crop_w], np.float32)
        image = image[:,:,[2, 1, 0]]
        image = image.transpose((2, 0, 1))
        if self.is_mirror:
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]
        return image.copy(),label.copy()

