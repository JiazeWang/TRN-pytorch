import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint
import torch

import random


class VideoRecord(object):
    def __init__(self, row, number):
        #self._data = row
        row = row.strip()
        row = row.split(' ')
        self.root = row[0]
        self.shot = row[1].split(',')
        self.frames = row[2].split(',')
        self.labelnew = row[3]
        self.num_selects = 8
        self.num_shots = len(self.shot)
        tick = ((self.num_shots)-1) / float(self.num_selects)
        self.offsets = int(random.randint(0,int(tick/2)) + tick * number)
        self.offsets_1 = int(tick * (number+1))
        self.select = random.randint(self.offsets, self.offsets_1)

    @property
    def path(self):
        return self.root+self.shot[self.select]

    @property
    def num_frames(self):
        return int(self.frames[self.select])

    @property
    def label(self):
        return self.labelnew


class TSNDataSetMovie(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=3, new_length=1, modality='RGB',
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(directory, self.image_tmpl.format('y', idx))).convert('L')
            return [x_img, y_img]

    def _parse_list(self):
        self.video_list0 = [VideoRecord(x, 0) for x in open(self.list_file)]
        self.video_list1 = [VideoRecord(x, 1) for x in open(self.list_file)]
        self.video_list2 = [VideoRecord(x, 2) for x in open(self.list_file)]

        self.video_list3 = [VideoRecord(x, 3) for x in open(self.list_file)]
        self.video_list4 = [VideoRecord(x, 4) for x in open(self.list_file)]
        self.video_list5 = [VideoRecord(x, 5) for x in open(self.list_file)]
        self.video_list6 = [VideoRecord(x, 6) for x in open(self.list_file)]
        self.video_list7 = [VideoRecord(x, 7) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """
        :param record: VideoRecord
        :return: list
        """

        average_duration = (record.num_frames - self.new_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration, size=self.num_segments)
        elif record.num_frames > self.num_segments:
            offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_val_indices(self, record):
        if record.num_frames > self.num_segments + self.new_length - 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
        else:
            offsets = np.zeros((self.num_segments,))
        return offsets + 1

    def _get_test_indices(self, record):

        tick = (record.num_frames - self.new_length + 1) / float(self.num_segments)

        offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])

        return offsets + 1

    def __getitem__(self, index):
        data = []
        data0, label0 = self.get_item_once(index,self.video_list0)
        data1, label0 = self.get_item_once(index,self.video_list1)
        data2, label0 = self.get_item_once(index,self.video_list2)

        data3, label0 = self.get_item_once(index,self.video_list3)
        data4, label0 = self.get_item_once(index,self.video_list4)
        data5, label0 = self.get_item_once(index,self.video_list5)
        data6, label0 = self.get_item_once(index,self.video_list6)
        data7, label0 = self.get_item_once(index,self.video_list7)

        data.extend(data0)
        data.extend(data1)
        data.extend(data2)

        data.extend(data3)
        data.extend(data4)
        data.extend(data5)
        data.extend(data6)
        data.extend(data7)

        process_data = self.transform(data)
        return process_data, label0

    def get_item_once(self, index, list):
        record = list[index]
        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)


    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = (int(seg_ind)-1)*4
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1
        #process_data = self.transform(images)
        newlabel = []
        #for line in record.label:
        labels =([int(l) for l in record.label.split(',')])
        labelnew = np.zeros([21])
        for i in labels:
            labelnew[i] = 1
        return images, labelnew

    def __len__(self):
        return len(self.video_list0)
