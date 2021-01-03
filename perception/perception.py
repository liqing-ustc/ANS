from utils import SYMBOLS, IMG_TRANSFORM, IMG_DIR
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import trange, tqdm
import math
import numpy as np
from collections import Counter

class Perception(object):
    def __init__(self):
        super(Perception, self).__init__()
        self.n_class = len(SYMBOLS)
        self.model = SymbolNet(self.n_class)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.device = torch.device('cpu')
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()
    
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

    
    def __call__(self, img_seq):
        batch_size = img_seq.shape[0]
        seq_len = img_seq.shape[1]
        images = img_seq.reshape((-1, img_seq.shape[-3], img_seq.shape[-2], img_seq.shape[-1]))
        logits = self.model(images)
        logits = logits.reshape((batch_size, seq_len, -1))
        # probs = torch.sigmoid(logits)
        probs = nn.functional.softmax(logits, dim=-1)
        if self.model.training:
            m = Categorical(probs=probs)
            preds = m.sample()
        else:
            preds = torch.argmax(probs, -1)

        return preds, probs

    def check_accuarcy(self, dataset):
        from utils import ID2SYM
        symbols = [x[0].split('/')[0] for x in dataset]
        labels = [ID2SYM(x[1]) for x in dataset]
        acc = np.mean(np.array(symbols) == np.array(labels))
        print(acc)

    def learn(self, dataset, n_iters=100):
        batch_size = 512
        labels = [l for _, l in dataset]
        counts = Counter(labels)
        max_count = counts.most_common(1)[0][1]
        total_count = len(labels)
        
        # class_weights = np.array([(total_count - counts[i])/ max(counts[i],1) for i in range(self.n_class)], dtype=np.float32)
        class_weights = np.array([max_count/ max(counts[i],1) for i in range(self.n_class)], dtype=np.float32)
        class_weights = torch.from_numpy(class_weights).to(self.device)

        classes_valid = np.array([i for i in range(self.n_class) if counts[i] > 0])
        classes_valid = torch.from_numpy(classes_valid).to(self.device)

        classes_invalid = np.array([i for i in range(self.n_class) if counts[i] == 0])
        # classes_invalid = torch.from_numpy(classes_invalid).to(self.device)

        criterion = nn.CrossEntropyLoss(weight=class_weights)
        # criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights, reduction='none')

        n_epochs = int(math.ceil(batch_size * n_iters / len(dataset)))
        n_epochs = max(n_epochs, 5) # run at least 5 epochs
        print(n_epochs, "epochs, ", end='')
        dataset = ImageSet(dataset)
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                         shuffle=True, num_workers=8)
        for epoch in range(n_epochs):
            for img, label in train_dataloader:
                img = img.to(self.device)
                label = label.to(self.device)
                logit = self.model(img)
                # logit[:, classes_invalid] = float('-inf')
                # label = nn.functional.one_hot(label, num_classes=self.n_class).type_as(logit)
                loss = criterion(logit, label)
                # loss[:, classes_invalid] = 0.
                # loss = torch.index_select(loss, 1, classes_valid)
                # loss = loss.mean()
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
        img = self.img_transform(img)

        return img, label

    def __len__(self):
        return len(self.dataset)



