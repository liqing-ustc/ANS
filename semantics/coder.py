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
        self.logPosterior = logPosterior
    
    def __call__(self, *inputs):
        fn = self.fn
        for x in inputs:
            fn = fn(x)
        return fn

    def __eq__(self, prog):
        if isinstance(self.fn, int) and isinstance(prog.fn, int):
            return self.fn == prog.fn
        return self.prog == prog.prog

    def __str__(self):
        return "%s %s %.2f"%(str(self.fn) if isinstance(self.fn, int) else "fn", self.prog, math.exp(self.logPosterior))

class TaskWrapper(object):
    def __init__(self, idx, max_examples=50, min_examples=5):
        self.idx = idx
        self.examples = []
        self.programs_over_time = []
        self.max_examples = max_examples
        self.min_examples = min_examples
        self.solved = False
        self.best_program = None

    def update_examples(self, examples):
        # self.examples = (self.examples + examples)[-self.max_examples:]
        self.examples = examples

    def update_programs(self, programs):
        self.programs_over_time.append(programs)
        self.check_solved()
    
    def check_solved(self):
        if len(self.examples) > 20 and np.exp(self.programs_over_time[-1][0].logPosterior) > 0.5:
            print("Solved Task-%d: %s"%(self.idx, self.programs_over_time[-1][0]))
            self.solved = True
            self.best_program = self.programs_over_time[-1][0]
            self.best_program.logPosterior = 0.0
        # window_size = 2
        # if len(self.programs_over_time) >= window_size:
        #     ave_posterior = np.mean([np.exp(ps[0].logPosterior) for ps in self.programs_over_time[-window_size:]])
        #     if ave_posterior > 0.5:
        #         print("Solved Task-%d: %s"%(self.idx, self.programs_over_time[-1][0]))
        #         self.solved = True
        #         self.best_program = self.programs_over_time[-1][0]
        #         self.best_program.logPosterior = 0.0
    
    def make_task(self):
        examples = self.examples
        if self.solved or len(examples) == 0:
            return None
        arity = Counter([len(xs) for xs, _ in examples]).most_common(1)[0][0]
        task_type = arrow(*([tint]*(arity + 1)))
        examples = [x for x in examples if len(x[0]) == arity]
        if len(examples) < self.min_examples:
            return None
        return Task(str(self.idx), task_type, examples)


class DreamCoder(object):
    def __init__(self):
        args = commandlineArguments(
            enumerationTimeout=20, activation='tanh', iterations=1, recognitionTimeout=3600,
            a=3, maximumFrontier=10, topK=2, pseudoCounts=30.0,
            helmholtzRatio=0.5, structurePenalty=1.,
            CPUs=numberOfCPUs(),
            extras=list_options)

        args['noConsolidation'] = True
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
        
        self.grammar = baseGrammar
        # self.sym2prog = self._sample_programs(baseGrammar, n_prog_per_task=0)
        self.sym2prog = [[] for _ in range(NUM_TASKS)]
        self.train_args = args
        self.tasks = [TaskWrapper(i) for i in range(NUM_TASKS)]

    def __call__(self):
        return self.sym2prog

    def _sample_programs(self, grammar, n_prog_per_task=5):
        sym2prog = []
        for _ in range(NUM_TASKS):
            programs = []
            while len(programs) < n_prog_per_task:
                arity = random.randint(0,2)
                task_type = arrow(*([tint]*(arity + 1)))
                prog = grammar.sample(task_type, maximumDepth=3)
                prog = ProgramWrapper(prog, -10.0)
                if prog not in programs:
                    programs.append(prog)
            sym2prog.append(programs)
        return sym2prog

    def set_sym2prog(self):
        sym2prog = [[] for _ in range(NUM_TASKS)]
        for i, task in enumerate(self.tasks):
            if task.solved:
                sym2prog[i] = [task.best_program]
        self.sym2prog = sym2prog

    def learn(self, dataset):
        for task, exps in zip(self.tasks, dataset):
            task.update_examples(exps)
        tasks = [t.make_task() for t in self.tasks]
        tasks = [t for t in tasks if t is not None]
        n_solved = len(['' for t in self.tasks if t.solved])
        if len(tasks) == 0:
            print("No found tasks to learn. %d/%d tasks solved."%(n_solved, len(self.tasks)))
            for task in self.tasks:
                if task.solved:
                    progs = task.programs_over_time[-1]
                    print("Task-%d: %s"%(task.idx, progs[0]))
            self.set_sym2prog()
            return 
        print("Tasks: %d/%d/%d (total/solved/learn)."%(len(self.tasks), n_solved, len(tasks)))
        self._print_tasks(tasks)
        result = explorationCompression(self.grammar, tasks, **self.train_args)
        self.grammar = result.grammars[-1]

        self.set_sym2prog()
        sym2prog = self.sym2prog
        n_prog_enum = 10
        for frontier in result.taskSolutions.values():
            task_idx = int(frontier.task.name)
            progs = [ProgramWrapper(x.program, x.logPosterior) for x in frontier.entries[:n_prog_enum]]
            sym2prog[task_idx] = progs
        sym2prog = self._removeEquivalentPrograms(sym2prog)
        self.sym2prog = sym2prog
        self._print_learned_progs(sym2prog)
        for task, progs in zip(self.tasks, sym2prog):
            if len(progs) == 0:
                continue
            task.update_programs(progs)


    def _print_tasks(self, tasks):
        for task in tasks:
            print("Task-%s (%s), Samples: %3d"%(task.name, task.request, len(task.examples)))
    
    def _print_learned_progs(self, sym2prog, topk=1):
        for i, progs in enumerate(sym2prog):
            if len(progs) == 0: continue
            print("Task-%d: %s"%(i, progs[0]))

    def _removeEquivalentPrograms(self, sym2prog, dataset=None):
        programs = [(p, i) for i, sym_progs in enumerate(sym2prog) for p in sym_progs]
        programs = sorted(programs, key=lambda x: (-x[0].logPosterior, x[1]))
        programs_keep = []
        symbols_keep = []
        for prog, sym in programs:
            if prog not in programs_keep:
                programs_keep.append(prog)
                symbols_keep.append(sym)
            if dataset is not None:
                pass # TODO: implement the equivalence remove on a dataset
        sym2prog = [[] for _ in range(len(sym2prog))]
        for prog, sym in zip(programs_keep, symbols_keep):
            sym2prog[sym].append(prog)

        return sym2prog
