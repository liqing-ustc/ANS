from utils import SYMBOLS, IMG_TRANSFORM, IMG_DIR
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class Perception(object):
    def __init__(self):
        super(Perception, self).__init__()
        self.model = SymbolNet()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.device = torch.device('cpu')
    
    def train(self):
        self.model.train()
    
    def eval(self):
        self.model.eval()
    
    def to(self, device):
        self.model.to(device)
        self.device = device
    
    def __call__(self, img_seq):
        batch_size = img_seq.shape[0]
        seq_len = img_seq.shape[1]
        images = img_seq.reshape((-1, img_seq.shape[-3], img_seq.shape[-2], img_seq.shape[-1]))
        logits = self.model(images)
        logits = logits.reshape((batch_size, seq_len, -1))
        probs = nn.functional.softmax(logits, dim=-1)
        epsilon = 0.
        probs = probs * (1 - epsilon) + epsilon / (len(SYMBOLS) - 1) 
        if self.model.training:
            m = Categorical(probs=probs)
            preds = m.sample()
        else:
            preds = torch.argmax(probs, -1)

        return preds, probs

    def check_accuarcy(self, dataset):
        from utils import ID2SYM
        import numpy as np
        symbols = [x[0].split('/')[0] for x in dataset]
        labels = [ID2SYM[x[1]] for x in dataset]
        acc = np.mean(np.array(symbols) == np.array(labels))
        print(acc)

    def learn(self, dataset, n_iters=100):
        dataset = [(img, label) for img_seq, label_seq in dataset for img, label in zip(img_seq, label_seq)]
        # self.check_accuarcy(dataset)
        dataset = ImageSet(dataset)
        train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=512,
                         shuffle=True, num_workers=4)
        for _ in range(n_iters):
            img, label = next(iter(train_dataloader))
            img = img.to(self.device)
            label = label.to(self.device)
            logit = self.model(img)
            loss = self.criterion(logit, label)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
                

class SymbolNet(nn.Module):
    def __init__(self):
        super(SymbolNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride = 1, padding = 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(30976, 128)
        self.fc2 = nn.Linear(128, len(SYMBOLS) - 1) # the last symbol is NULL

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
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



