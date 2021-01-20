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

import resnet_scan
from utils import SYMBOLS, INP_VOCAB, DEVICE

class Encoder(nn.Module):
    def __init__(self, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
       
        self.image_encoder = resnet_scan.make_model(n_class=len(SYMBOLS))
        self.symbol_embeding = nn.Embedding(len(SYMBOLS), emb_dim)
        self.aux_embeding = nn.Embedding(3, emb_dim) # input embedding for START, END, NULL
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, bidirectional=True)
        
    def forward(self, images, inp_lens):
        #src = [src len, batch size]
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
        outputs, (hidden, cell) = self.rnn(inputs)
        
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = 13
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout, bidirectional=True)
        
        self.fc_out = nn.Linear(hid_dim * 2, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell



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
        hidden, cell = self.encoder(src, inp_lens)
        
        #first input to the decoder is the <sos> tokens
        input = trg[:, 0]
        outputs = []
        for t in range(1, trg.shape[1]):
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            teacher_force = random.random() < teacher_forcing_ratio
            input = trg[:, t] if teacher_force else output.argmax(1) 

            outputs.append(output)
        
        outputs = torch.stack(outputs).transpose(0, 1) #[batch, seq, prob]
        return outputs


def make_model(config=None):
    OUTPUT_DIM = 13
    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 2
    ENC_DROPOUT = 0.5
    DEC_DROPOUT = 0.5

    enc = Encoder(ENC_EMB_DIM, HID_DIM, N_LAYERS, ENC_DROPOUT)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, HID_DIM, N_LAYERS, DEC_DROPOUT)

    model = Seq2Seq(enc, dec)
    return model