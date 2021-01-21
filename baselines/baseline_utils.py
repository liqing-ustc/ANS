import sys
sys.path.append("../")
from utils import *
from dataset import HINT, HINT_collate

INP_VOCAB = SYMBOLS + [START, END, NULL]
RES_VOCAB = DIGITS + [START, END, NULL]

RES_MAX_LEN = 10

reverse = True
def res2seq(res, pad=True):
    seq = [list(str(r)) for r in res]
    if reverse:
        seq = [s[::-1] for s in seq]
    seq = [[START] + s + [END] for s in seq]
    if pad:
        max_len = max([len(s) for s in seq])
        seq = [s + [NULL]*(max_len - len(s)) for s in seq]
    seq = [list(map(RES_VOCAB.index, s)) for s in seq]
    return seq

def seq2res(seq):
    seq = [RES_VOCAB[x] for x in seq]
    seq = [x for x in seq if x in DIGITS]
    if reverse:
        seq = seq[::-1]
    res = int(''.join(seq)) if len(seq) > 0 else -1
    return res
