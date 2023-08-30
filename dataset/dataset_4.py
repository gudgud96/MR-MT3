import torch
from torch.utils.data import Dataset, DataLoader

# import tensorflow as tf
# tf.config.set_visible_devices([], 'GPU')

import os
from itertools import cycle
import json
import random
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import librosa
import note_seq
from glob import glob
from contrib import event_codec, note_sequences, spectrograms, vocabularies, run_length_encoding, metrics_utils
from contrib.preprocessor import slakh_class_to_program_and_is_drum, add_track_to_notesequence, PitchBendError
from tqdm import tqdm


class SlakhDatasetEncodecDump(Dataset):

    def __init__(
        self, 
        root_dir, 
        random_select=True
    ) -> None:
        super().__init__()
        self.files = sorted(glob(os.path.join(root_dir, "*")))
        self.random_select = random_select


    def __getitem__(self, idx):
        fname = self.files[idx]
        inputs_npy = np.load(os.path.join(fname, "inputs.npy"))
        targets_npy = np.load(os.path.join(fname, "targets.npy"))
        
        if self.random_select:
            length_idx = np.arange(inputs_npy.shape[0])
            np.random.shuffle(length_idx)
            indices = length_idx[:2]
            inputs_npy = inputs_npy[indices]
            targets_npy = targets_npy[indices]

        return torch.from_numpy(inputs_npy), torch.from_numpy(targets_npy), fname
    
    def __len__(self):
        return len(self.files)


def collate_fn(lst):
    inputs = [k[0] for k in lst]
    targets = [k[1] for k in lst]
    return torch.cat(inputs), torch.cat(targets)

if __name__ == '__main__':
    dataset = SlakhDatasetEncodecDump(
        root_dir='/home/gudgud96/encodec_mt3/test/',
    )
    print(len(dataset))
    # print("pitch", dataset.codec.event_type_range("pitch"))
    # print("velocity", dataset.codec.event_type_range("velocity"))
    # print("tie", dataset.codec.event_type_range("tie"))
    # print("program", dataset.codec.event_type_range("program"))
    # print("drum", dataset.codec.event_type_range("drum"))
    # dl = DataLoader(dataset, batch_size=2, num_workers=0, collate_fn=collate_fn, shuffle=False)
    # for idx, item in enumerate(dl):
    #     inputs, targets = item
    #     print(idx, inputs.shape, targets.shape)
    #     if idx > 10:
    #         break
    