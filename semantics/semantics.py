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
import torch

from dreamcoder.dreamcoder import commandlineArguments, explorationCompression
from dreamcoder.utilities import eprint, flatten, testTrainSplit, numberOfCPUs
from dreamcoder.grammar import Grammar
from dreamcoder.task import Task
from dreamcoder.type import Context, arrow, tbool, tlist, tint, t0, UnificationFailure
from dreamcoder.recognition import RecurrentFeatureExtractor
from dreamcoder.program import Program, Invented
from dreamcoder.frontier import Frontier, FrontierEntry

from dreamcoder.domains.hint.hintPrimitives import McCarthyPrimitives
from dreamcoder.domains.hint.main import main, list_options, LearnedFeatureExtractor

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
        self.cache = {} # used for fast computation
    
    def __call__(self, *inputs):
        if len(inputs) != self.arity or None in inputs:
            return None
        if inputs in self.cache:
            return self.cache[inputs]
        fn = self.fn
        try:
            for x in inputs:
                fn = fn(x)
        except RecursionError as e:
            fn = None
        self.cache[inputs] = fn
        return fn

    def __eq__(self, prog): # only used for removing equivalent semantics
        if self.arity != prog.arity:
            return False
        if isinstance(self.fn, int) and isinstance(prog.fn, int):
            return self.fn == prog.fn
        # if self.y is not None and prog.y is not None:
        #     assert len(self.y) == len(prog.y) # the program should be evaluated on same examples
        #     return np.mean(self.y[self.y!=None] == prog.y[self.y!=None]) > 0.95
        # return self.prog == prog.prog
        return False

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

    def evaluate(self, examples, store_y=True): 
        y = np.array([self(*xs) for xs in examples])
        if store_y: # if store_y, save y for equivalence checking
            self.y = y
        return y

def compute_likelihood(program=None, examples=None):
    if program is None or examples is None:
        return 0.
    else:
        pred = program.evaluate([e[0] for e in examples], store_y=False)
        gt = np.array([e[1] for e in examples])
        return np.mean(pred == gt)

class Semantics(object):
    def __init__(self, idx, program=None):
        self.idx = idx
        self.examples = None
        self.program = program
        self.arity = None
        self.solved = False
        self.likelihood = 0.

    def update_examples(self, examples):
        if not examples:
            self.examples = []
            return
        arity = Counter([len(x[0]) for x in examples]).most_common(1)[0][0]
        examples = [x[:2] for x in examples if len(x[0]) == arity] 

        self.arity = arity
        self.examples = examples
        self.likelihood = compute_likelihood(self.program, self.examples)
        self.check_solved()

    def update_program(self, entry):
        program = ProgramWrapper(entry.program)
        likelihood = compute_likelihood(program, self.examples)
        if (likelihood > self.likelihood) or \
            (likelihood == self.likelihood and len(str(program)) < len(str(self.program))):
            self.program = program
            self.likelihood = likelihood
            self.check_solved()
    
    def check_solved(self):
        if self.arity == 0 and self.likelihood > 0.:
            self.solved = True
        elif self.arity > 0 and self.likelihood >= 0.9 and len(set(self.examples)) >= 80:
            self.solved = True
        else:
            self.solved = False

    def __call__(self, *inputs):
        return self.program(*inputs)

    def make_task(self):
        min_examples = 10 
        max_samples = 100
        # if len(self.examples) < min_examples or (self.arity == 0 and self.solved):
        if len(self.examples) < min_examples or self.solved:
            return None
        task_type = arrow(*([tint]*(self.arity + 1)))
        examples = self.examples
        if len(examples) > max_samples:
            examples = random.sample(self.examples, k=max_samples)
        return Task(str(self.idx), task_type, examples)

    def clear(self):
        self.examples = None
        self.program = None
        self.arity = None
        self.solved = False
        self.likelihood = 0.
    
    def save(self):
        model = {'idx': self.idx, 'solved': self.solved, 'likelihood': self.likelihood, 'arity': self.arity}
        model['program'] = None if self.program is None else self.program.prog
        return model

    def load(self, model):
        self.idx = model['idx']
        self.solved = model['solved']
        self.likelihood = model['likelihood']
        self.arity = model['arity']
        self.program = None if model['program'] is None else ProgramWrapper(Program.parse(model['program']))

class DreamCoder(object):
    def __init__(self):
        args = commandlineArguments(
            enumerationTimeout=200, activation='tanh', iterations=1, recognitionTimeout=3600,
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
        
        self.primitives = McCarthyPrimitives()
        baseGrammar = Grammar.uniform(self.primitives)
        self.grammar = baseGrammar
        self.train_args = args
        self.semantics = [Semantics(i) for i in range(len(SYMBOLS))] 
        self.allFrontiers = None
        self.helmholtzFrontiers = None

    def __call__(self):
        return self.semantics

    def save(self):
        model = [smt.save() for smt in self.semantics]
        return model

    def load(self, model):
        if model is None:
            return
        assert len(self.semantics) == len(model)
        for i in range(len(self.semantics)):
            self.semantics[i].load(model[i])

    def rescore_frontiers(self, tasks):
        if self.allFrontiers is None:
            return
        print('Rescoring %d frontiers...'%len(self.allFrontiers))
        id2task = {t.name: t for t in tasks}
        id2frontier = {f.task.name: f for f in self.allFrontiers}
        allFrontiers = {}
        for name in id2task.keys():
            task = id2task[name]
            examples = task.examples
            if name not in id2frontier:
                frontier = Frontier([], task=task)
            else:
                frontier = id2frontier[name]
                frontier.task = task
                for entry in frontier.entries:
                    program = ProgramWrapper(entry.program)
                    entry.logLikelihood = float(np.log(compute_likelihood(program=program, examples=examples)))
                    entry.logPosterior = entry.logLikelihood + entry.logPrior
                frontier.removeLowLikelihood(low=0.1)

            allFrontiers[task] = frontier
        self.allFrontiers = allFrontiers

    def learn(self, dataset):
        tasks = []
        max_arity = 0
        for smt, exps in zip(self.semantics, dataset):
            smt.update_examples(exps)
            t = smt.make_task()
            if t is not None:
                tasks.append(t)
                max_arity = max(smt.arity, max_arity)
        self.train_args['enumerationTimeout'] = 10 if max_arity == 0 else 200
        # self.train_args['iterations'] = 1 if max_arity == 0 else 3
        n_solved = len(['' for t in self.semantics if t.solved])
        print("Semantics: %d/%d/%d (total/solved/learn)."%(len(self.semantics), n_solved, len(tasks)))
        if len(tasks) == 0:
            self._print_semantics()
            return 
        self._print_tasks(tasks)
        self.update_grammar()
        print(self.grammar)
        # print(self.allFrontiers)
        self.rescore_frontiers(tasks)
        if self.allFrontiers is not None:
            print(self.allFrontiers.values())

        if self.helmholtzFrontiers is not None:
            requests_old ={x.task.request for x in self.helmholtzFrontiers()}
            requests = {t.request for t in tasks}
            # if new requests, discard old helmholtz frontiers
            if requests != requests_old:
                self.helmholtzFrontiers = None

        result = explorationCompression(self.grammar, tasks, allFrontiers=self.allFrontiers, helmholtzFrontiers=self.helmholtzFrontiers, **self.train_args)
        self.allFrontiers = list(result.allFrontiers.values())
        self.helmholtzFrontiers = result.helmholtzFrontiers

        for frontier in result.taskSolutions.values():
            if not frontier.entries: continue
            symbol_idx = int(frontier.task.name)
            # print(frontier)
            self.semantics[symbol_idx].update_program(frontier.bestPosterior)
        # examples = [xs for t in tasks for xs, y in t.examples]
        # self._removeEquivalentSemantics(examples)
        self._removeEquivalentSemantics()
        self._print_semantics()
        # self.grammar = result.grammars[-1]

    def update_grammar(self):
        programs = [Invented(smt.program.prog_ori) for smt in self.semantics 
            if smt.solved and smt.program.arity > 0 and '#' not in str(smt.program)]
            # if '#' in the program, the program uses a invented primitive, it is very likely to have a high computation cost.
            # Therefore we don't add this program into primitives, since it might slow the enumeration a lot.
            # it might be resolved by increasing the enumeration time
        new_grammar = Grammar.uniform(self.primitives + programs)
        # self.train_args['enumerationTimeout'] += 100 * len(programs)
        if new_grammar != self.grammar:
            self.grammar = new_grammar
            self.helmholtzFrontiers = None
            self.allFrontiers = None
            print("Update grammar with invented programs and set frontiers to none.")
        

    def _print_semantics(self):
        for smt in sorted(self.semantics, key=lambda x: x.idx):
            print("Symbol-%02d: %s %.2f"%(smt.idx, smt.program, smt.likelihood))
            # print("Solved!" if smt.solved else "")

    def _print_tasks(self, tasks):
        for task in tasks:
            print("Symbol-%02d (%s), Samples: %3d, "%(int(task.name), task.request, len(task.examples)), Counter(task.examples).most_common(10))

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
                    if (len(smt_i.examples) * smt_i.likelihood) >= (len(smt_j.examples) * smt_j.likelihood):
                        smt_j.clear()
                    else:
                        smt_i.clear()
                        break
