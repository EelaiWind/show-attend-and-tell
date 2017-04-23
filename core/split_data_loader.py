from __future__ import print_function

import numpy as np
from numpy.random import RandomState
import cPickle as pickle
import hickle
import sys
import os
import random 

class SplitDataLoader(object):
    def __init__(self, root_path, dataset_directoris, batch_size):
        self.m_dataset_count = len(dataset_directoris)
        self.m_dataset_path_prefix = [ os.path.join(root_path, category, category+".") for category in dataset_directoris ]
        self.m_dataset_size = []
        self.m_total_dataset_size = 0
        for prefix in self.m_dataset_path_prefix:
            filename = prefix + "image.idxs.pkl"
            with open(filename, 'rb') as f:
                size = len(pickle.load(f))
            assert(size >= batch_size)
            print("%s dataset has %d data" % (filename, size))
            self.m_dataset_size.append(size)
            self.m_total_dataset_size += size

        self.m_batch_size = batch_size
        self.reset_whole_dataset()

    def reset_whole_dataset(self):
        self.m_is_epoch_finish = False
        self.m_feature = None
        self.m_caption = None
        self.m_image_idxs = None
        self.m_ground_truths = None
        self.m_is_current_dataset_end = True
        self.m_current_dataset_position = 0
        self.m_current_dataset_id = -1

    def _open_new_dataset_and_shuffle(self, prefix):
        with open(prefix + 'captions.pkl', 'rb') as f:
            self.m_caption = pickle.load(f)
        with open(prefix + 'image.idxs.pkl', 'rb') as f:
            self.m_image_idxs = pickle.load(f)
        self.m_feature = hickle.load(prefix + 'features.hkl')

        self.m_current_dataset_position = 0
        self.m_is_current_dataset_end = False

        seed = random.randint(0,  2**32-1)
        prng = RandomState(seed)
        prng.shuffle(self.m_caption)
        prng = RandomState(seed)
        prng.shuffle(self.m_image_idxs)

    def get_next_batch(self):
        if self.m_is_epoch_finish == True:
            print("[Warning] All epoch is over but reset_whole_dataset() was not called", file=sys.stderr)

        if self.m_is_current_dataset_end == True:
            self.m_feature = None
            self.m_caption = None
            self.m_image_idxs = None
            self.m_ground_truths = None

            self.m_current_dataset_id = 0 if self.m_current_dataset_id == self.m_dataset_count-1 else (self.m_current_dataset_id +1)

            prefix = self.m_dataset_path_prefix[self.m_current_dataset_id]
            self._open_new_dataset_and_shuffle(prefix)
        
        dataset_size = self.m_dataset_size[self.m_current_dataset_id]
        if self.m_current_dataset_position + self.m_batch_size >= dataset_size:
            self.m_current_dataset_position = dataset_size - self.m_batch_size
            self.m_is_current_dataset_end = True
            self.m_is_epoch_finish |= (self.m_current_dataset_id == self.m_dataset_count-1)

        end_index = self.m_current_dataset_position + self.m_batch_size
        batch_caption = self.m_caption[self.m_current_dataset_position : end_index]
        batch_image_idxs = self.m_image_idxs[self.m_current_dataset_position : end_index]
        batch_feature = self.m_feature[batch_image_idxs]
        self.m_current_dataset_position = end_index
        
        # all captions of the first image_id in this batch
        self.m_ground_truths = self.m_caption[self.m_image_idxs == batch_image_idxs[0]]

        return batch_feature, batch_caption

    def is_epoch_finish(self):
        return self.m_is_epoch_finish

    def get_total_dataset_size(self):
        return self.m_total_dataset_size

    def get_groud_truths(self):
        return self.m_ground_truths







