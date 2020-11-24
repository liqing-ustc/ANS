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
            self.fn = prog.evaluate([])
        except RecursionError as e:
            self.fn = None
        self.prog = str(prog)
        self.arity = len(prog.infer().functionArguments())
        self.logPosterior = logPosterior
        self._name = None
        self.y = None # used for equivalence check
    
    def __call__(self, *inputs):
        if len(inputs) != self.arity:
            return None
        fn = self.fn
        for x in inputs:
            fn = fn(x)
        return fn

    def __eq__(self, prog):
        if isinstance(self.fn, int) and isinstance(prog.fn, int):
            return self.fn == prog.fn
        if self.y is not None and prog.y is not None:
            return np.all(self.y == prog.y)
        return self.prog == prog.prog

    def __str__(self):
        return "%s %s %.2f"%(self.name, self.prog, math.exp(self.logPosterior))

    @property
    def name(self):
        if self._name is not None: return self._name
        if isinstance(self.fn, int):
            self._name = str(self.fn)
        else:
            self._name = "fn"
            pass # TODO: assign name based on the function
        return self._name


    def evaluate(self, examples): # used for equivalence check on a dataset
        self.y = np.array([self(*xs) for xs in examples])

class Semantics(object):
    def __init__(self, idx, min_examples=10):
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
            # solved_threhold = 300
        if posterior > 0.95 and len(self.examples) * posterior > solved_threhold: # more careful!
            self.solved = True
            self.program.logPosterior = 0.0 
    
    def __call__(self, *inputs):
        return self.program(*inputs)

    def make_task(self):
        examples = self.examples
        if self.solved or len(examples) == 0:
            return None
        arity = Counter([len(x[0]) for x in examples]).most_common(1)[0][0]
        task_type = arrow(*([tint]*(arity + 1)))
        examples = [x for x in examples if len(x[0]) == arity]
        if len(examples) < self.min_examples:
            return None

        if arity > 0:
            examples = sorted(examples, key=lambda x: -x[2])
            print(examples[:5])
            print(examples[-5:])

        counts = {}
        for e in examples:
            p = np.exp(e[2])
            e = e[:2]
            if e not in counts:
                counts[e] = 0.
            counts[e] += p
        n_examples = len(examples)
        Z = sum(list(counts.values()))
        examples = []
        for e, p in sorted(counts.items(), key=lambda x: -x[1]):
            examples.extend([e] * int(p / Z * n_examples))
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

        args['noConsolidation'] = True
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
        print("Semantics: %d/%d/%d (total/solved/learn)."%(len(self.semantics), n_solved, len(tasks)))
        if len(tasks) == 0:
            self._print_semantics()
            return 
        self._print_tasks(tasks)
        result = explorationCompression(self.grammar, tasks, **self.train_args)
        # self.grammar = result.grammars[-1]
        print(self.grammar)

        programs = [(smt.idx, smt.program) for smt in self.semantics if smt.solved]
        for frontier in result.taskSolutions.values():
            if not frontier.entries: continue
            symbol_idx = int(frontier.task.name)
            best_entry = frontier.bestPosterior
            prog = ProgramWrapper(best_entry.program, best_entry.logPosterior)
            programs.append((symbol_idx, prog))
        examples = [xs for t in tasks for xs, y in t.examples]
        programs = self._removeEquivalent(programs, examples)

        # clear all past programs
        for smt in self.semantics:
            smt.program = None

        for idx, p in programs:
            smt = self.semantics[idx]
            smt.update_program(p)

        self._print_semantics()

    def _print_semantics(self):
        for smt in sorted(self.semantics, key=lambda x: str(x.program)):
            print("Symbol-%02d: %s"%(smt.idx, smt.program))
            # print("Solved!" if smt.solved else "")

    def _print_tasks(self, tasks):
        for task in tasks:
            # print("Symbol-%s (%s), Samples: %3d, "%(task.name, task.request, len(task.examples)), task.examples[:20])
            print("Symbol-%02d (%s), Samples: %3d, "%(int(task.name), task.request, len(task.examples)), Counter(task.examples))

        json.dump([t.examples for t in tasks], open('outputs/tasks.json', 'w'))

    def _removeEquivalent(self, programs, examples=None):
        programs = sorted(programs, key=lambda x: (-x[1].logPosterior, x[0]))
        if examples is not None:
            examples = list(set(examples))
            for _, p in programs:
                p.evaluate(examples)
        programs_keep = []
        symbols_keep = []
        for i, p in programs:
            if p not in programs_keep:
                programs_keep.append(p)
                symbols_keep.append(i)
        programs = list(zip(symbols_keep, programs_keep))
        return programs
