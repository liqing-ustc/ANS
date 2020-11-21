from utils import PROGRAMS
from .semantics import DreamCoder

class SemanticsGT():
    def __init__(self):
        self.semantics = PROGRAMS

    def __call__(self):
        return self.semantics

    def train(self):
        pass

    def eval(self):
        pass


def build(config=None):
    # model = SemanticsGT()
    model = DreamCoder()
    return model