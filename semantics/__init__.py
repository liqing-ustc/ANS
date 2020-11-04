from utils import PROGRAMS

class SemanticsGT():
    def __init__(self):
        pass

    def __call__(self):
        return PROGRAMS

    def train(self):
        pass

    def eval(self):
        pass


def build(config=None):
    model = SemanticsGT()
    return model