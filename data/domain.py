DIGITS= [str(i) for i in range(0, 10)]
OPERATORS = list('+-*/!')
NULL = '<NULL>'
SYMBOLS = DIGITS + OPERATORS + [NULL]
SYM2ID = {v:i for i, v in enumerate(SYMBOLS)}
ID2SYM = {i:v for i, v in enumerate(SYMBOLS)}
MAX_RES = 1e3

import math
from inspect import signature
class Program():
    def __init__(self, fn):
        self.fn = fn
        self.arity = len(signature(fn).parameters)

    def __call__(self, *inputs):
        res = self.fn(*inputs)
        if res is not None and res <= MAX_RES:
            return res
        return None

functions = [
    lambda: 0, lambda: 1, lambda: 2, lambda: 3, lambda: 4, lambda: 5, lambda: 6, lambda: 7, lambda: 8, lambda: 9,
    lambda x,y: x+y, lambda x,y: max(0, x-y), lambda x,y: x*y, lambda x,y: x//y, lambda x: math.factorial(x) if x <=20 else None,
    lambda: None
]

PROGRAMS = [Program(f) for f in functions]
SYM2PROG= {s:p for s, p in zip(SYMBOLS, PROGRAMS)}