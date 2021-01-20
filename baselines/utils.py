# import argparse
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.optim.lr_scheduler import StepLR

# import os
# from PIL import Image
# from copy import deepcopy
# from tqdm import tqdm
# import time

import torch
import numpy as np
np.set_printoptions(precision=2, suppress=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ROOT_DIR = './data/'
IMG_DIR = ROOT_DIR + 'symbol_images/'
IMG_SIZE = 32

from torchvision import transforms
IMG_TRANSFORM = transforms.Compose([
                    transforms.CenterCrop(IMG_SIZE),
                    transforms.ToTensor(),
                    # transforms.Lambda(lambda x: 1. - x),
                    # transforms.Normalize((0.5,), (1,))
                ])

from PIL import Image, ImageOps
def pad_image(img, desired_size, fill=0):
    delta_w = desired_size - img.size[0]
    delta_h = desired_size - img.size[1]
    padding = (delta_w//2, delta_h//2, delta_w-(delta_w//2), delta_h-(delta_h//2))
    new_img = ImageOps.expand(img, padding, fill)
    return new_img

DIGITS= [str(i) for i in range(0, 10)]
OPERATORS = list('+-*/')
PARENTHESES = list('()')
SYMBOLS = DIGITS + OPERATORS + PARENTHESES
SYM2ID = lambda x: SYMBOLS.index(x)
ID2SYM = lambda x: SYMBOLS[x]

NULL = '<NULL>'
START = '<START>'
END = '<END>'

INP_VOCAB = SYMBOLS + [START, END, NULL]
RES_VOCAB = DIGITS + [START, END, NULL]

reverse = True
def res2seq(res):
    seq = str(res)
    if reverse:
        seq = seq[::-1]
    return [RES_VOCAB.index(x) for x in [START] + list(seq) + [END]]

def seq2res(seq):
    seq = [RES_VOCAB[x] for x in seq]
    seq = [x for x in seq if x in DIGITS]
    if reverse:
        seq = seq[::-1]
    res = int(''.join(seq)) if len(seq) > 0 else -1
    return res
