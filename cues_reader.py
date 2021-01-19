import os
import os.path as osp

import numpy as np
import torch
# import cPickle
import pickle

class CuesReader(object):
    def __init__(self, cues_dir, cues_name):
        self.cues_dir = cues_dir
        self.cues_name = cues_name
        with open(osp.join(cues_dir, cues_name), 'rb') as handle:
            self.cues_data = pickle.load(handle)
        # self.cues_data = pickle.load(open(osp.join(cues_dir, cues_name)))

    def get_forward(self, image_id):

        labels = np.zeros((1, 1, 21), dtype=int)
        cues = np.zeros((21, 41, 41), dtype=float)
        labels_i = self.cues_data['%s_labels' % image_id]
        labels[0, 0, 0] = 1.0
        labels[0, 0, labels_i] = 1.0

        cues_i = self.cues_data['%s_cues' % image_id]
        cues[cues_i[0], cues_i[1], cues_i[2]] = 1.0

        return labels, cues




