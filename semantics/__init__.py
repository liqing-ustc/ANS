from utils import SYMBOLS, SYM2PROG, NULL
from .semantics import DreamCoder

class SemanticsGT():
    def __init__(self):
        self.semantics = [SYM2PROG[s] for s in SYMBOLS if s != NULL]

    def __call__(self):
        return self.semantics

    def save(self):
        pass

    def load(self, model):
        pass


def build(config=None):
    if config.semantics:
        model = SemanticsGT()
    else:
        model = DreamCoder()
    return model