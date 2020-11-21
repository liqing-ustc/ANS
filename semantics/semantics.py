import sys
sys.setrecursionlimit(int(1e4))
sys.path.insert(0, "./semantics/dreamcoder")

import random
from collections import defaultdict, Counter 
import json
import math
import os
import datetime
import numpy as np

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

class ProgramWrapper(object):
    def __init__(self, prog, logPosterior=0.):
        try:
            self.fn = prog.uncurry().evaluate([])
        except RecursionError as e:
            self.fn = None
        self.prog = str(prog)
        self.arity = len(prog.infer().functionArguments())
        self.logPosterior = logPosterior
    
    def __call__(self, *inputs):
        fn = self.fn
        for x in inputs:
            fn = fn(x)
        if not isinstance(fn, int):
            raise ValueError
        return fn

    def __eq__(self, prog):
        if isinstance(self.fn, int) and isinstance(prog.fn, int):
            return self.fn == prog.fn
        return self.prog == prog.prog

    def __str__(self):
        return "%s %s %.2f"%(str(self.fn) if isinstance(self.fn, int) else "fn", self.prog, math.exp(self.logPosterior))

class Semantics(object):
    def __init__(self, idx, min_examples=20):
        self.idx = idx
        self.examples = []
        self.program = None
        self.min_examples = min_examples
        self.solved = False

    def update_examples(self, examples):
        self.examples = examples

    def update_program(self, program):
        self.program = program
        self.check_solved()
    
    def check_solved(self):
        posterior = np.exp(self.program.logPosterior)
        if self.program.arity == 0:
            solved_threhold = 50
        else:
            solved_threhold = float("inf")
        if len(self.examples) * posterior > solved_threhold: # more careful!
            self.solved = True
            self.program.logPosterior = 0.0 
    
    def __call__(self, *inputs):
        return self.program(*inputs)

    def make_task(self):
        examples = self.examples
        if self.solved or len(examples) == 0:
            return None
        arity = Counter([len(xs) for xs, _ in examples]).most_common(1)[0][0]
        task_type = arrow(*([tint]*(arity + 1)))
        examples = [x for x in examples if len(x[0]) == arity]
        if len(examples) < self.min_examples:
            return None
        self.examples = examples
        return Task(str(self.idx), task_type, examples)


class DreamCoder(object):
    def __init__(self):
        args = commandlineArguments(
            enumerationTimeout=30, activation='tanh', iterations=1, recognitionTimeout=3600,
            a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
            helmholtzRatio=0.5, structurePenalty=1.,
            CPUs=numberOfCPUs(),
            extras=list_options)

        # args['noConsolidation'] = True
        random.seed(args.pop("random_seed"))
        args['contextual'] = True

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
        
        self.grammar = baseGrammar
        self.train_args = args
        self.semantics = [Semantics(i) for i in range(NUM_TASKS)]

    def __call__(self):
        return self.semantics

    # def _sample_programs(self, grammar, n_prog_per_task=5):
    #     sym2prog = []
    #     for _ in range(NUM_TASKS):
    #         programs = []
    #         while len(programs) < n_prog_per_task:
    #             arity = random.randint(0,2)
    #             task_type = arrow(*([tint]*(arity + 1)))
    #             prog = grammar.sample(task_type, maximumDepth=3)
    #             prog = ProgramWrapper(prog, -10.0)
    #             if prog not in programs:
    #                 programs.append(prog)
    #         sym2prog.append(programs)
    #     return sym2prog

    def learn(self, dataset):
        for smt, exps in zip(self.semantics, dataset):
            smt.update_examples(exps)
        tasks = [t.make_task() for t in self.semantics]
        tasks = [t for t in tasks if t is not None]
        n_solved = len(['' for t in self.semantics if t.solved])
        if len(tasks) == 0:
            print("No found semantics to learn. %d/%d semantics solved."%(n_solved, len(self.semantics)))
            for smt in self.semantics:
                if smt.solved:
                    print("Symbol-%d: %s"%(smt.idx, smt.program))
            return 
        print("Semantics: %d/%d/%d (total/solved/learn)."%(len(self.semantics), n_solved, len(tasks)))
        self._print_tasks(tasks)
        result = explorationCompression(self.grammar, tasks, **self.train_args)
        # self.grammar = result.grammars[-1]
        print(self.grammar)

        programs = [(smt.idx, smt.program) for smt in self.semantics if smt.solved]
        for frontier in result.taskSolutions.values():
            symbol_idx = int(frontier.task.name)
            best_entry = frontier.bestPosterior
            prog = ProgramWrapper(best_entry.program, best_entry.logPosterior)
            programs.append((symbol_idx, prog))
        programs = self._removeEquivalent(programs)
        for idx, p in programs:
            smt = self.semantics[idx]
            if smt.solved:
                continue
            print("Symbol-%d: %s "%(idx, p), end="")
            smt.update_program(p)
            print("Solved!" if smt.solved else "")

    def _print_tasks(self, tasks):
        for task in tasks:
            # print("Symbol-%s (%s), Samples: %3d, "%(task.name, task.request, len(task.examples)), task.examples[:20])
            print("Symbol-%s (%s), Samples: %3d, "%(task.name, task.request, len(task.examples)), Counter(task.examples))

        with open('outputs/tasks.json', 'w') as f:
            for task in tasks:
                f.write(json.dumps(task.examples)+'\n')

    def _removeEquivalent(self, programs, dataset=None):
        programs = sorted(programs, key=lambda x: (-x[1].logPosterior, x[0]))
        programs_keep = []
        symbols_keep = []
        for i, p in programs:
            if p not in programs_keep:
                programs_keep.append(p)
                symbols_keep.append(i)
            if dataset is not None:
                pass # TODO: implement the equivalence remove on a dataset
        programs = list(zip(symbols_keep, programs_keep))
        return programs
