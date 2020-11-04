from utils import SYM2ID, ROOT_DIR, IMG_DIR, NULL, IMG_TRANSFORM
from copy import deepcopy
import random
import json
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate

class HINT(Dataset):
    def __init__(self, split='train', numSamples=None, randomSeed=None):
        super(HINT, self).__init__()
        
        assert split in ['train', 'val', 'test']
        self.split = split
        self.dataset = json.load(open(ROOT_DIR + 'expr_%s.json'%split))
        if numSamples:
            if randomSeed:
                random.seed(randomSeed)
                random.shuffle(self.dataset)
            self.dataset = self.dataset[:numSamples]
            
        for x in self.dataset:
            x['len'] = len(x['expr'])
        
        self.img_transform = IMG_TRANSFORM

    def __getitem__(self, index):
        sample = deepcopy(self.dataset[index])
        img_seq = []
        for img_path in sample['img_paths']:
            img = Image.open(IMG_DIR+img_path).convert('L')
            #print(img.size, img.mode)
            img = self.img_transform(img)
            img_seq.append(img)
        # del sample['img_paths']
        sample['expr'] = ''.join(sample['expr'])
        
        label_seq = [SYM2ID[sym] for sym in sample['expr']]
        sample['img_seq'] = img_seq
        sample['label_seq'] = label_seq
        return sample
            
    
    def __len__(self):
        return len(self.dataset)

    def filter_by_len(self, max_len):
        self.dataset = [x for x in self.dataset if x['len'] <= max_len]


def HINT_collate(batch):
    max_len = np.max([x['len'] for x in batch])
    zero_img = torch.zeros_like(batch[0]['img_seq'][0])
    img_paths_list = []
    head_list = []
    for sample in batch:
        sample['img_seq'] += [zero_img] * (max_len - sample['len'])
        sample['img_seq'] = torch.stack(sample['img_seq'])
        
        sample['label_seq'] += [SYM2ID[NULL]] * (max_len - sample['len'])
        sample['label_seq'] = torch.tensor(sample['label_seq'])

        # sample['head'] += [-2] * (max_len - sample['len'])
        # sample['head'] = torch.tensor(sample['head'])
        img_paths_list.append(sample['img_paths'])
        del sample['img_paths']

        head_list.append(sample['head'])
        del sample['head']
        
    batch = default_collate(batch)
    batch['img_paths'] = img_paths_list
    batch['head'] = head_list
    return batch

if __name__ == '__main__':
    val_set = HINT('val')
    val_loader = DataLoader(val_set, batch_size=32,
                         shuffle=True, num_workers=4, collate_fn=HINT_collate)

    print(next(iter(val_loader)))