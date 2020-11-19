import sys
sys.setrecursionlimit(int(1e4))
sys.path.insert(0, "./semantics/dreamcoder")

import random
from collections import defaultdict, Counter 
import json
import math
import os
import datetime

from dreamcoder.dreamcoder import explorationCompression
from dreamcoder.utilities import eprint, flatten, testTrainSplit
from dreamcoder.grammar import Grammar
from dreamcoder.task import Task
from dreamcoder.type import Context, arrow, tbool, tlist, tint, t0, UnificationFailure
from dreamcoder.domains.hint.hintPrimitives import McCarthyPrimitives
from dreamcoder.recognition import RecurrentFeatureExtractor
from dreamcoder.domains.hint.makeTasks import make_list_bootstrap_tasks

from dreamcoder.domains.hint.main import main, list_options, LearnedFeatureExtractor
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.utilities import numberOfCPUs

from utils import SYMBOLS

NUM_TASKS = len(SYMBOLS) - 1

def make_tasks():
    tasks = []
    for i in range(len(SYMBOLS) - 1):
        tasks.append(Task("symbol-%d"%i, ))

class ProgramWrapper(object):
    def __init__(self, prog):
        try:
            self.fn = prog.uncurry().evaluate([])
        except RecursionError as e:
            self.fn = None
        self.prog = str(prog)
    
    def __call__(self, *inputs):
        fn = self.fn
        for x in inputs:
            fn = fn(x)
        return fn

    def __eq__(self, prog):
        if isinstance(self.fn, int) and isinstance(prog.fn, int):
            return self.fn == prog.fn
        return self.prog == prog.prog

class DreamCoder(object):
    def __init__(self):
        args = commandlineArguments(
            enumerationTimeout=50, activation='tanh', iterations=10, recognitionTimeout=3600,
            a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
            helmholtzRatio=0.5, structurePenalty=1.,
            CPUs=numberOfCPUs(),
            extras=list_options)

        """
        Takes the return value of the `commandlineArguments()` function as input and
        trains/tests the model on manipulating sequences of numbers.
        """
        random.seed(args.pop("random_seed"))

        baseGrammar = Grammar.uniform(McCarthyPrimitives())

        extractor = {
            "learned": LearnedFeatureExtractor,
        }[args.pop("extractor")]
        extractor.H = args.pop("hidden")

        timestamp = datetime.datetime.now().isoformat()
        outputDirectory = "outputs/%s"%timestamp
        os.system("mkdir -p %s"%outputDirectory)
        
        args.update({
            "featureExtractor": extractor,
            "outputPrefix": "%s/list"%outputDirectory,
            "evaluationTimeout": 0.0005,
        })
        args.pop("maxTasks")
        args.pop("split")
        args['iterations'] = 1
        
        self.grammar = baseGrammar
        self.sym2prog = self._sample_programs(baseGrammar)
        self.train_args = args

    def __call__(self):
        return self.sym2prog

    def _sample_programs(self, grammar, n_prog_per_task=5):
        sym2prog = []
        for _ in range(NUM_TASKS):
            programs = []
            while len(programs) < n_prog_per_task:
            # for _ in range(n_prog_per_task):
                arity = random.randint(0,2)
                task_type = arrow(*([tint]*(arity + 1)))
                prog = grammar.sample(task_type, maximumDepth=3)
                prog = ProgramWrapper(prog)
                if prog not in programs:
                    programs.append(prog)
            sym2prog.append(programs)
        return sym2prog

    def _make_tasks(self, dataset):
        tasks = []
        for task_idx in range(NUM_TASKS):
            examples = dataset[task_idx]
            if len(examples) == 0:
                continue
            arity = Counter([len(xs) for xs, _ in examples]).most_common(1)[0][0]
            task_type = arrow(*([tint]*(arity + 1)))
            examples = [x for x in examples if len(x[0]) == arity]
            tasks.append(Task(str(task_idx), task_type, examples))
        return tasks

    def learn(self, dataset):
        tasks = self._make_tasks(dataset)
        # print(tasks)
        result = explorationCompression(self.grammar, tasks, **self.train_args)
        self.grammar = result.grammars[-1]

        n_prog_sampling = 5
        n_prog_enum = 10
        sym2prog = self._sample_programs(self.grammar, n_prog_sampling)
        for frontier in result.taskSolutions.values():
            task_idx = int(frontier.task.name)
            progs = [ProgramWrapper(x.program) for x in frontier.entries[:n_prog_enum]]
            sym2prog[task_idx] = progs + sym2prog[task_idx]
        self.sym2prog = sym2prog

