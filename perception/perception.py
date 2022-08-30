from utils import SYMBOLS, IMG_TRANSFORM, IMG_DIR, pad_image, SYM2ID
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image, ImageOps
from tqdm import trange, tqdm
import math
import numpy as np
from collections import Counter
from . import resnet_scan, lenet_scan
from torchvision import transforms
import random

tok_convert = {'*': 'times', '/': 'div', 'a': 'alpha', 'b': 'beta', 'c': 'gamma', 'd': 'phi', 'e': 'theta'}
tok_convert = {v:k for k, v in tok_convert.items()}
def check_accuarcy(dataset):
    symbols = [x[0].split('/')[0] for x in dataset]
    symbols = [tok_convert.get(x, x) for x in symbols]
    symbols = [SYM2ID(x) for x in symbols]
    labels = [x[1] for x in dataset]
    acc = np.mean(np.array(symbols) == np.array(labels))
    print(acc, end=', ')

class Perception(object):
    def __init__(self):
        super(Perception, self).__init__()
        self.n_class = len(SYMBOLS)
        self.model = SentenceEncoder(self.n_class)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.device = torch.device('cpu')
        self.training = False
        self.min_examples = 200
        self.selflabel_dataset = None
    
    def train(self):
        # self.model.train()
        self.model.eval()
        self.training = True

    def eval(self):
        self.model.eval()
        self.training = False
    
    def to(self, device):
        self.model.to(device)
        self.device = device

    def save(self, save_optimizer=True):
        saved = {'model': self.model.state_dict()}
        if save_optimizer:
            saved['optimizer'] = self.optimizer.state_dict()
        return saved
    
    def load(self, loaded, image_encoder_only=False):
        if image_encoder_only:
            self.model.image_encoder.load_state_dict(loaded['model'])
        else:
            self.model.load_state_dict(loaded['model'])
        if 'optimizer' in loaded:
            self.optimizer.load_state_dict(loaded['optimizer'])

    def extend(self, n):
        self.n_class += n
        self.model.extend(n)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def selflabel(self, dataset):
        symbols = [(x, SYM2ID(y)) for sample in dataset for x, y in zip(sample['img_paths'], sample['expr'])]
        dataloader = torch.utils.data.DataLoader(ImageSet(symbols), batch_size=512,
                         shuffle=False, drop_last=False, num_workers=8)
        with torch.no_grad():
            self.eval()
            prob_all = []
            for img, _ in dataloader:
                img = img.to(self.device)
                prob = self.model.image_encoder(img)
                prob = nn.functional.softmax(prob, dim=-1)
                prob_all.append(prob)
            prob_all = torch.cat(prob_all)
        
        confidence = 0.95
        selflabel_dataset = {}
        probs, preds = torch.max(prob_all, dim=1)
        probs = probs.cpu().numpy()
        preds = preds.cpu().numpy()
        for cls_id in range(self.n_class):
            idx_list = np.where(preds == cls_id)[0]
            idx_list = sorted(idx_list, key=lambda x: probs[x], reverse=True)
            idx_list = [i for i in idx_list if probs[i] >= confidence]
            images = [symbols[i][0] for i in idx_list]
            # images = list(set(images))
            labels = [symbols[i][1] for i in idx_list]
            acc = np.mean(np.array(labels) == cls_id)
            selflabel_dataset[cls_id] = [(x, cls_id) for x in images]
            print("Add %d samples for class %d, acc %.2f."%(len(images), cls_id, acc))
        img2cls = {img: cls_id for examples in selflabel_dataset.values() for img, cls_id in examples}
        dataset = [(sample['img_paths'], list(map(lambda x: img2cls.get(x, None), sample['img_paths']))) for sample in dataset]
        dataset = [(x, y) for x, y in dataset if None not in y]
        self.selflabel_dataset = dataset
        print(f'Add {len(dataset)} self-labelled examples for perception.')
        self.learn([], n_iters=1000)


    
    def __call__(self, images, src_len):
        logits = self.model(images, src_len)
        # probs = torch.sigmoid(logits)
        probs = nn.functional.softmax(logits, dim=-1)
        if self.training:
            m = Categorical(probs=probs)
            preds = m.sample()
        else:
            preds = torch.argmax(probs, -1)

        return preds, probs


    def learn(self, dataset=[], n_iters=100):
        dataset = dataset + self.selflabel_dataset
        batch_size = 32
        criterion = nn.CrossEntropyLoss()
        # criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights, reduction='none')

        n_epochs = int(math.ceil(batch_size * n_iters / len(dataset)))
        print(n_epochs, "epochs, ", end='')
        dataset = ImageSeqSet(dataset)
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                        shuffle=True, collate_fn=collate, num_workers=8)
        self.model.train()
        for epoch in range(n_epochs):
            for sample in train_dataloader:
                img_seq = sample['img_seq'].to(self.device)
                sentence = sample['sentence'].to(self.device)
                length = sample['length']
                logit = self.model(img_seq, length)
                # label = nn.functional.one_hot(label, num_classes=self.n_class).type_as(logit)
                loss = criterion(logit, sentence)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                

class SymbolNet(nn.Module):
    def __init__(self, n_class):
        super(SymbolNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(6, 16, 3, stride = 1, padding = 1)
        self.fc1 = nn.Linear(16 * 8 * 8, 120) 
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_class)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SentenceEncoder(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        self.image_encoder = resnet_scan.make_model(n_class)
        input_dim, emb_dim, hidden_dim, layers, dropout = 512, 128, 128, 2, 0.5
        self.n_token = n_class + 3
        self.embedding = nn.Embedding(self.n_token, emb_dim)
        self.fc_in = nn.Linear(input_dim, emb_dim)
        self.encoder = nn.LSTM(emb_dim, hidden_dim, layers, dropout=dropout, bidirectional=True, batch_first=True)
        self.fc_out = nn.Linear(2 * hidden_dim, n_class)
    
    def forward(self, src, src_len):
        src = self.image_encoder.backbone(src)
        src = self.fc_in(src)

        max_len = src_len.max()
        current = 0
        padded_src = []
        emb_start = self.embedding(torch.tensor([self.n_token - 3]).to(src.device))
        emb_end = self.embedding(torch.tensor([self.n_token - 2]).to(src.device))
        emb_null = self.embedding(torch.tensor([self.n_token - 1]).to(src.device))
        for l in src_len:
            current_input = src[current:current+l]
            current_input = [emb_start, current_input, emb_end] + [emb_null] * (max_len - l) 
            current_input = torch.cat(current_input)
            padded_src.append(current_input)
            current += l
        src = torch.stack(padded_src)

        outputs, _ = self.encoder(src)
        logits = self.fc_out(outputs)
        unroll_logits = [p[1:l+1] for l, p in zip(src_len, logits)] # the first token is START
        logits = torch.cat(unroll_logits)
        return logits

class ImageSet(Dataset):
    def __init__(self, dataset):
        super(ImageSet, self).__init__()
        self.dataset = dataset
        self.img_transform = IMG_TRANSFORM

    def __getitem__(self, index):
        sample = self.dataset[index]
        img_path, label = sample
        img = Image.open(IMG_DIR+img_path).convert('L')
        img = ImageOps.invert(img)
        img = pad_image(img, 60)
        img = transforms.functional.resize(img, 40)
        img = self.img_transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)

class ImageSeqSet(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset
        self.img_transform = IMG_TRANSFORM

    def __getitem__(self, index):
        sample = self.dataset[index]
        img_paths, labels = sample
        images = []
        for img_path in img_paths:
            img = Image.open(IMG_DIR+img_path).convert('L')
            img = ImageOps.invert(img)
            img = pad_image(img, 60)
            img = transforms.functional.resize(img, 40)
            img = self.img_transform(img)
            images.append(img)

        return {'img_seq': images, 'sentence': labels}

    def __len__(self):
        return len(self.dataset)
    

def collate(batch):
    img_seq_list = []
    sentence_list = []
    length_list = []
    for sample in batch:
        img_seq_list.extend(sample['img_seq'])
        sentence_list.extend(sample['sentence'])
        length_list.append(len(sample['sentence']))

        del sample['img_seq']
        del sample['sentence']
    
    batch = {}
    batch['img_seq'] = torch.stack(img_seq_list)
    batch['sentence'] = torch.tensor(sentence_list)
    batch['length'] = torch.tensor(length_list)
    return batch



