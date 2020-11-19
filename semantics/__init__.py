from utils import PROGRAMS
from .coder import DreamCoder

class SemanticsGT():
    def __init__(self):
        self.sym2prog = [[x] for x in PROGRAMS]

    def __call__(self):
        return self.sym2prog

    def train(self):
        pass

    def eval(self):
        pass


def build(config=None):
    # model = SemanticsGT()
    model = DreamCoder()
    return model