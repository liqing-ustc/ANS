from dataset import HINT, HINT_collate

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import random
import math
import time
from tqdm import tqdm

import sys
sys.path.append("..")
from perception import resnet_scan
from baseline_utils import SYMBOLS, INP_VOCAB, RES_VOCAB, DEVICE

class Encoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
       
        self.image_encoder = resnet_scan.make_model(n_class=len(SYMBOLS))
        self.symbol_embeding = nn.Embedding(len(SYMBOLS), emb_dim)
        self.aux_embeding = nn.Embedding(3, emb_dim) # input embedding for START, END, NULL
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=True)
        
    def forward(self, images, inp_lens):
        logits = self.image_encoder(images)
        probs = F.softmax(logits, dim=-1)

        max_len = inp_lens.max()
        inputs = torch.matmul(probs, self.symbol_embeding.weight)
        current = 0
        padded_inputs = []
        emb_start = self.aux_embeding(torch.tensor([0]).to(DEVICE))
        emb_end = self.aux_embeding(torch.tensor([1]).to(DEVICE))
        emb_null = self.aux_embeding(torch.tensor([2]).to(DEVICE))
        for l in inp_lens:
            current_input = inputs[current:current+l]
            current_input = [emb_start, current_input, emb_end] + [emb_null] * (max_len - l) 
            current_input = torch.cat(current_input)
            padded_inputs.append(current_input)
            current += l
        inputs = torch.stack(padded_inputs)
        inputs = inputs.transpose(0, 1)
        output, hidden = self.rnn(inputs)

        hidden = hidden.view(self.n_layers, -1, *hidden.shape[1:])
        hidden = hidden[-1]
        hidden = hidden.transpose(0, 1)
        hidden = hidden.contiguous().view(hidden.shape[0], -1)
        
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.rnn = nn.GRU(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=False)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        output, hidden = self.rnn(embedded, hidden)
        output = F.relu(output.squeeze(0))
        output = self.fc_out(output)
        return output, hidden



class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, src, trg, inp_lens, teacher_forcing_ratio = 1.):
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        #tensor to store decoder outputs
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden = self.encoder(src, inp_lens)
        
        #first input to the decoder is the <sos> tokens
        input = trg[:, 0]
        hidden = torch.stack([hidden] * self.decoder.n_layers)
        outputs = []
        for t in range(1, trg.shape[1]):
            output, hidden = self.decoder(input, hidden)
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else output.argmax(1) 

            outputs.append(output)
        
        outputs = torch.stack(outputs).transpose(0, 1) #[batch, seq, prob]
        return outputs


def make_model(config=None):
    ENC_LAYERS = config.enc_layers
    DEC_LAYERS = config.dec_layers
    EMB_DIM = 128
    HID_DIM = 128
    DROPOUT = 0.5
    OUTPUT_DIM = len(RES_VOCAB)

    enc = Encoder(EMB_DIM, HID_DIM, ENC_LAYERS, DROPOUT)
    dec = Decoder(OUTPUT_DIM, EMB_DIM, HID_DIM * 2, DEC_LAYERS, DROPOUT)

    model = Seq2Seq(enc, dec)
    return model