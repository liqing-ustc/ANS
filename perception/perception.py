from utils import SYMBOLS, IMG_TRANSFORM, IMG_DIR, pad_image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageOps
from tqdm import trange, tqdm
import math
import numpy as np
from collections import Counter
from . import resnet_scan, lenet_scan
from torchvision import transforms
import random

tok_convert = {'*': 'times', '/': 'div', 'a': 'alpha', 'b': 'beta', 'g': 'gamma', 't': 'theta', 'p': 'phi'}
tok_convert = {v:k for k, v in tok_convert.items()}
def check_accuarcy(dataset):
    from utils import SYM2ID
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
        # self.model = SymbolNet(self.n_class)
        self.model = resnet_scan.make_model(self.n_class)
        # self.model = lenet_scan.make_model(self.n_class)
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
    
    def load(self, loaded):
        self.model.load_state_dict(loaded['model'])
        if 'optimizer' in loaded:
            self.optimizer.load_state_dict(loaded['optimizer'])

    def extend(self, n):
        self.n_class += n
        self.model.extend(n)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

    def selflabel(self, symbols):
        dataloader = torch.utils.data.DataLoader(ImageSet(symbols), batch_size=512,
                         shuffle=False, drop_last=False, num_workers=8)
        with torch.no_grad():
            self.eval()
            prob_all = []
            for img, _ in dataloader:
                img = img.to(self.device)
                prob = self.model(img)
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
        self.selflabel_dataset = selflabel_dataset


    
    def __call__(self, images):
        logits = self.model(images)
        # probs = torch.sigmoid(logits)
        probs = nn.functional.softmax(logits, dim=-1)
        if self.training:
            m = Categorical(probs=probs)
            preds = m.sample()
        else:
            preds = torch.argmax(probs, -1)

        return preds, probs


    def learn(self, dataset, n_iters=100):
        batch_size = 512
        labels = [l for i, l in dataset]
        counts = Counter(labels)

        check_accuarcy(dataset)
        classes_invalid = [i for i in range(self.n_class) if counts[i] < self.min_examples]
        if classes_invalid and self.selflabel_dataset is not None:
            for cls_id in classes_invalid:
                dataset.extend(random.choices(self.selflabel_dataset[cls_id], k=self.min_examples - counts[cls_id]))
            check_accuarcy(dataset)
            print(len(dataset), end=', ')

        labels = [l for i, l in dataset]
        counts = Counter(labels)
        max_count = counts.most_common(1)[0][1]
        total_count = len(labels)
        class_weights = np.array([(total_count - counts[i])/ max(counts[i],1) for i in range(self.n_class)], dtype=np.float32)
        class_weights = np.array([max_count/ max(counts[i],1) for i in range(self.n_class)], dtype=np.float32)
        class_weights = torch.from_numpy(class_weights).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        # criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights, reduction='none')

        n_epochs = int(math.ceil(batch_size * n_iters / len(dataset)))
        print(n_epochs, "epochs, ", end='')
        dataset = ImageSet(dataset)
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                         shuffle=True, num_workers=8)
        self.model.train()
        for epoch in range(n_epochs):
            for img, label in train_dataloader:
                img = img.to(self.device)
                label = label.to(self.device)
                logit = self.model(img)
                # label = nn.functional.one_hot(label, num_classes=self.n_class).type_as(logit)
                loss = criterion(logit, label)
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



