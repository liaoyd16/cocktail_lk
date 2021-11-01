###
# Editor: Kai Li
# Date: 2021-06-24 19:01:48
# LastEditors: Kai Li
# LastEditTime: 2021-06-26 17:40:22
# Description: file content
###

import os

# PROJ_ROOT = "/Users/*/*/cocktail"
PROJ_ROOT = "/home/liaoyuanda/cocktail/"
DATA_ROOT = os.path.join(PROJ_ROOT, "dataset/")

def audio_name_(speaker, iblock):
    return os.path.join(DATA_ROOT, "sp{}/{}.wav".format(speaker, iblock))

dump_meta = {
    'dump_layer': 'y5',
}

data_meta = {
    'speakers': [0,1,2,3,4],
    'using_speakers': [1,3], # must be even to make target-masker pairs
    'pairs': 1,
    'cut_secs': 0.5,
    'feature_block_per_speaker': [(0, 0),(0, 0),(0, 0),(0, 0),(0, 0)],
    'train_blocks_per_speaker': [i for i in range(27)],
    'test_blocks_per_speaker': [i for i in range(27,29)],
    'slices_per_block': 120, # each wav length=60, 60 / 0.5 = 120
    'Fs': 22050,
    'slice_len': 11025, # 22050 * cut_secs
    'specgram_size': (256, 128),
    'fft_size': 2048,
    'step_size': 128,
    'spec_thresh': 0,
}

model_meta = {
    'feature_vector_size': 256,
    'feature_net_classes': 5,
    'attend_layers': 4, ##
    'layer_attentions': [1,1,1,1], ##
    'activation_size': {
        'x0': [1,  256,128],
        'x1': [8,  256,128],
        'x2': [16, 128,64],
        'x3': [32, 64, 32],
        'x4': [64, 32, 16],
        'x5': [128,16, 8],
        'x6': [4096],
        'x7': [256]
    },
    'rf_expand': { # plus for expansion, minus for pooling
        1: [1, 1, 1],
        2: [-2, 1, 1, 1],
        3: [-2, 1, 1, 1],
        4: [-2, 1, 1, 1],
        5: [-2, 1, 1, 1],
        6: [-2, 1, 1, 1],
    },
    'POW': 2,
    'fake_features': True,
    'embedder': {
        'sr': 16000,
        'window': 0.025,
        'hop': 0.01,
        'nmels': 40,
        'hidden': 768,
        'num_layer': 3,
        'proj': 256,
        'tisv_frame': 180,
        'nfft': 512,
    }
}

assert(len(model_meta['layer_attentions']) == model_meta['attend_layers'])

import torch
import numpy as np

EPS = 1e-5

def lg(x):
    x[x <= 0] = EPS
    return np.log(x) / np.log(10)

def mel(specgram):
    specgram[specgram < 0] = 0
    return lg(1 + specgram/ 4)

DEVICE_ID = "cuda:0"
device = torch.device(DEVICE_ID if torch.cuda.is_available() else "cpu")
assert(len(data_meta['using_speakers']) % 2 == 0)
