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
from dreamcoder.program import Program, Invented

from dreamcoder.domains.hint.main import main, list_options, LearnedFeatureExtractor
from dreamcoder.dreamcoder import commandlineArguments
from dreamcoder.utilities import numberOfCPUs

from utils import SYMBOLS

class ProgramWrapper(object):
    def __init__(self, prog):
        try:
            self.fn = prog.evaluate([])
        except RecursionError as e:
            self.fn = None
        self.prog_ori = prog
        self.prog = str(prog)
        self.arity = len(prog.infer().functionArguments())
        self._name = None
        self.y = None # used for equivalence check
    
    def __call__(self, *inputs):
        if len(inputs) != self.arity:
            return None
        fn = self.fn
        try:
            for x in inputs:
                fn = fn(x)
            return fn
        except RecursionError as e:
            return None

    def __eq__(self, prog): # only used for removing equivalent semantics
        if self.arity != prog.arity:
            return False
        if isinstance(self.fn, int) and isinstance(prog.fn, int):
            return self.fn == prog.fn
        if self.y is not None and prog.y is not None:
            assert len(self.y) == len(prog.y) # the program should be evaluated on same examples
            return np.mean(self.y == prog.y) > 0.95
        return self.prog == prog.prog

    def __str__(self):
        return "%s %s"%(self.name, self.prog)

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
        examples = [xs for xs in examples if len(xs) == self.arity]
        self.y = np.array([self(*xs) for xs in examples])
        return self.y

class Semantics(object):
    def __init__(self, idx):
        self.idx = idx
        self.examples = None
        self.program = None
        self.arity = None
        self.solved = False
        self.likelihood = 0.
        self.total_examples = 0

    def update_examples(self, examples):
        if not examples:
            return
        arity = Counter([len(x[0]) for x in examples]).most_common(1)[0][0]
        examples = [x for x in examples if len(x[0]) == arity]

        counts = {}
        T = 1 / 5
        for e in examples:
            p = np.exp(e[2])
            xs, y = e[:2]
            if xs not in counts:
                counts[xs] = {}
            if y not in counts[xs]:
                counts[xs][y] = np.array([0., 0.])
            counts[xs][y] += np.array([p ** T, 1])
        new_counts = []
        for xs, y2p in counts.items():
            y2p = sorted(y2p.items(), key=lambda x: -x[1][0])
            y, p = y2p[0]
            new_counts.append(((xs, y), p))
        total_examples = int(sum([p[1] for e, p in new_counts]))
        counts = [(e, p[0]) for e, p in new_counts]
        Z = sum([p for e, p in counts])
        counts = [(e, p/Z) for e, p in counts]
        counts = sorted(counts, key=lambda x: -x[1])

        if total_examples < 10:
            self.clear() # clear the semantics
        else:
            self.total_examples = total_examples
            self.arity = arity
            self.examples = counts
            self.check_solved()
            # print(self.examples)

    def update_program(self, entry):
        if math.exp(entry.logLikelihood) > self.likelihood:
            self.program = ProgramWrapper(entry.program)
            self.check_solved()
        # if self.arity > 0 and not self.program.prog.startswith("(lambda " * self.arity + "(fix"):
        #     self.clear()
    
    def check_solved(self):
        self.update_likelihood()
        solved = False
        if self.likelihood >= 0.9:
            if self.arity == 0:
                solved = True
            else:
                # check the number of distinct examples
                if self.program.prog.startswith("(lambda " * self.arity + "(fix"):
                    if self.arity == 1 and len(self.examples) >= 5:
                        solved = True
                    elif self.arity == 2 and len(self.examples) >= 100:
                        solved = True 
        if solved:
            self.solved = True
            self.likelihood = 1.0
            if self.arity > 0:
                print(len(self.examples), sorted([x for x, p in self.examples]))
    
    def update_likelihood(self):
        if self.program is None:
            self.likelihood = 0.
        else:
            pred = self.program.evaluate([e[0] for e, p in self.examples])
            gt = [e[1] for e, p in self.examples]
            self.likelihood = np.sum(np.array(pred == gt) * np.array([p for e, p in self.examples]))

    @property
    def priority(self):
        # used for abduction, favor the solved semantics a little more
        return self.likelihood + (1.0 if self.solved else 0.)
    
    def __call__(self, *inputs):
        return self.program(*inputs)

    def make_task(self):
        if self.solved or self.total_examples == 0:
            return None
        task_type = arrow(*([tint]*(self.arity + 1)))

        examples = []
        n_examples = min(self.total_examples, 100)
        # examples = random.choices([e for e, _ in self.examples], weights=[p for _, p in self.examples], k=n_examples)
        for e, p in self.examples:
            examples.extend([e] * int(math.ceil(p * n_examples)))
        examples = random.sample(examples, n_examples)
        return Task(str(self.idx), task_type, examples)

    def clear(self):
        self.examples = None
        self.program = None
        self.arity = None
        self.solved = False
        self.likelihood = 0.
        self.total_examples = 0
    
    def save(self):
        model = {'idx': self.idx, 'solved': self.solved, 'likelihood': self.likelihood, 
                'total_examples': self.total_examples, 'arity': self.arity}
        model['program'] = None if self.program is None else self.program.prog
        return model

    def load(self, model):
        self.idx = model['idx']
        self.solved = model['solved']
        self.likelihood = model['likelihood']
        self.total_examples = model['total_examples']
        self.arity = model['arity']
        self.program = None if model['program'] is None else ProgramWrapper(Program.parse(model['program']))

class DreamCoder(object):
    def __init__(self):
        args = commandlineArguments(
            enumerationTimeout=200, activation='tanh', iterations=3, recognitionTimeout=3600,
            a=3, maximumFrontier=5, topK=2, pseudoCounts=30.0,
            helmholtzRatio=0.5, structurePenalty=1.,
            CPUs=min(numberOfCPUs(), 8),
            extras=list_options)

        args['noConsolidation'] = True
        args.pop("random_seed")
        args['contextual'] = True
        args['biasOptimal'] = True
        args['auxiliaryLoss'] = True
        args['activation'] = "relu"
        args['useDSL'] = False


        extractor = {
            "learned": LearnedFeatureExtractor,
        }[args.pop("extractor")]
        extractor.H = args.pop("hidden")

        timestamp = datetime.datetime.now().isoformat()
        outputDirectory = "tmp/%s"%timestamp
        os.system("mkdir -p %s"%outputDirectory)
        
        args.update({
            "featureExtractor": extractor,
            "outputPrefix": "%s/hint"%outputDirectory,
            "evaluationTimeout": 0.0005,
        })
        args.pop("maxTasks")
        args.pop("split")
        
        baseGrammar = Grammar.uniform(McCarthyPrimitives())
        self.grammar = baseGrammar
        self.train_args = args
        self.semantics = [Semantics(i) for i in range(len(SYMBOLS) - 1)] # one symbol is NULL

    def __call__(self):
        return self.semantics

    def save(self):
        model = [smt.save() for smt in self.semantics]
        return model

    def load(self, model):
        assert len(self.semantics) == len(model)
        for i in range(len(self.semantics)):
            self.semantics[i].load(model[i])

    def learn(self, dataset):
        tasks = []
        max_arity = 0
        for smt, exps in zip(self.semantics, dataset):
            if smt.solved: continue
            smt.update_examples(exps)
            t = smt.make_task()
            if t is not None:
                tasks.append(t)
                max_arity = max(smt.arity, max_arity)
        self.train_args['enumerationTimeout'] = 10 if max_arity == 0 else 200
        self.train_args['iterations'] = 1 if max_arity == 0 else 3
        n_solved = len(['' for t in self.semantics if t.solved])
        print("Semantics: %d/%d/%d (total/solved/learn)."%(len(self.semantics), n_solved, len(tasks)))
        if len(tasks) == 0:
            self._print_semantics()
            return 
        self._print_tasks(tasks)
        self.update_grammar()
        print(self.grammar)
        result = explorationCompression(self.grammar, tasks, **self.train_args)

        for frontier in result.taskSolutions.values():
            if not frontier.entries: continue
            symbol_idx = int(frontier.task.name)
            self.semantics[symbol_idx].update_program(frontier.bestPosterior)
        # examples = [xs for t in tasks for xs, y in t.examples]
        # self._removeEquivalentSemantics(examples)
        self._removeEquivalentSemantics()
        self._print_semantics()
        # self.grammar = result.grammars[-1]

    def update_grammar(self):
        programs = [Invented(smt.program.prog_ori) for smt in self.semantics if smt.solved and smt.program.arity > 0]
        self.grammar = Grammar.uniform(McCarthyPrimitives() + programs)
        

    def _print_semantics(self):
        for smt in sorted(self.semantics, key=lambda x: str(x.program)):
            print("Symbol-%02d: %s %.2f"%(smt.idx, smt.program, smt.likelihood))
            # print("Solved!" if smt.solved else "")

    def _print_tasks(self, tasks):
        for task in tasks:
            # print("Symbol-%s (%s), Samples: %3d, "%(task.name, task.request, len(task.examples)), task.examples[:20])
            print("Symbol-%02d (%s), Samples: %3d, "%(int(task.name), task.request, len(task.examples)), Counter(task.examples))

        json.dump([t.examples for t in tasks], open('outputs/tasks.json', 'w'))

    def _removeEquivalentSemantics(self, examples=None):
        if examples is not None:
            examples = list(set(examples))
            for smt in self.semantics:
                if smt.program is not None:
                    smt.program.evaluate(examples)
        
        for i in range(len(self.semantics) - 1):
            smt_i = self.semantics[i]
            if smt_i.program is None:
                continue
            for j in range(i+1, len(self.semantics)):
                smt_j = self.semantics[j]
                if smt_j.program is None:
                    continue
                if smt_i.program == smt_j.program:
                    if smt_i.likelihood >= smt_j.likelihood:
                        smt_j.clear()
                    else:
                        smt_i.clear()
                        break
