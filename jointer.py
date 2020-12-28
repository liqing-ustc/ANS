import perception, syntax, semantics
import numpy as np
from copy import deepcopy
import sys
from func_timeout import func_timeout, FunctionTimedOut
from utils import SYMBOLS, DEVICE
from collections import Counter
from time import time
import torch

class Node:
    def __init__(self, symbol, smt):
        self.symbol = symbol
        self.smt = smt
        self.children = []
        self._res = None

    def res(self):
        if self._res is not None:
            return self._res

        self._res = self.smt(*[x.res() for x in self.children])
        if self._res is None or self._res > sys.maxsize:
            self._res = None
        return self._res

    def children_res_valid(self):
        for ch in self.children:
            if ch._res is None: 
                return False
        return True

def joint_prob(sentence, transitions, semantics, sent_probs, trans_probs):
    probs = [sent_probs[i,w] for i, w in enumerate(sentence)] + \
            [trans_probs[i][t] for i, t in enumerate(transitions)] + \
            [semantics[w].likelihood for w in sentence]
    probs = np.array(probs) + 1e-12
    log_prob = np.log(probs).sum()
    return log_prob

class AST: # Abstract Syntax Tree
    def __init__(self, sentence, dependencies, semantics, transitions=None, sent_probs=None, transition_probs=None):
        self.sentence = sentence
        self.dependencies = dependencies
        self.transitions = transitions
        self.sent_probs = sent_probs
        self.transition_probs = transition_probs
        self.semantics = semantics
        self.joint_prob = None

        nodes = [Node(s, semantics[s]) for s in sentence]
        for node, h in zip(nodes, dependencies):
            if h == -1:
                root_node = node
                continue
            nodes[h].children.append(node)
        self.nodes = nodes
        self.root_node = root_node

        self._res = None
        try:
            # TODO: set a timeout for the execution
            # self._res = func_timeout(timeout=0.01, func=root_node.res)
            self._res = root_node.res()
        except (IndexError, TypeError, ZeroDivisionError, ValueError, RecursionError, FunctionTimedOut) as e:
            # Must be extremely careful about these errors
            # if isinstance(e, FunctionTimedOut):
            #     print(e)
            pass

    def res(self): return self._res

    def res_all(self): return [nd.res() for nd in self.nodes]

    def abduce(self, y, module=None):
        if self._res is not None and self._res == y:
            self.joint_prob = joint_prob(self.sentence, self.transitions, self.semantics, self.sent_probs, self.transition_probs)
            return self
        
        if module == 'perception':
            et = self.abduce_perception(y)
            if et is not None:
                return et
        elif module == 'syntax':
            et = self.abduce_syntax(y)
            if et is not None:
                return et
        elif module == 'semantics':
            et = self.abduce_semantics(y)
            if et is not None:
                return et
        
        return None

        
    def abduce_semantics(self, y):
        # abduce over semantics
        # Currently, if the root node's children are valid, we directly change the result to y
        if self.root_node.children_res_valid():
            if self.root_node.smt.solved and self.root_node.smt.arity == 0:
                unsolveds = [smt.idx for smt in self.semantics if not smt.solved]
                if not unsolveds:
                    return None
                root_node_idx = self.dependencies.index(-1)
                root_node_probs = self.sent_probs[root_node_idx]
                sampling_probs = np.array([root_node_probs[i] for i in unsolveds])
                sym = unsolveds[np.argmax(sampling_probs)]
                self.root_node.symbol = sym
                self.sentence[root_node_idx] = sym
            self._res = y
            self.root_node._res = y
            self.joint_prob = joint_prob(self.sentence, self.transitions, self.semantics, self.sent_probs, self.transition_probs)
            return self
        return None

    def abduce_perception(self, y):
        epsilon = 0
        # abduce over sentence
        sent_pos_list = np.argsort([self.sent_probs[i, s] for i, s in enumerate(self.sentence)])
        for sent_pos in sent_pos_list:
            s_prob = self.sent_probs[sent_pos]
            for sym_pos in np.argsort(s_prob)[::-1]:
                if s_prob[sym_pos] <= epsilon:
                    break
                new_sentence = deepcopy(self.sentence)
                new_sentence[sent_pos] = sym_pos
                et = AST(new_sentence, self.dependencies, self.semantics)
                if et.res() is not None and et.res() == y:
                    et.joint_prob = joint_prob(new_sentence, self.transitions, self.semantics, self.sent_probs, self.transition_probs)
                    return et
        return None

    def abduce_syntax(self, y):
        # abduce over parse
        # if current trans is 'S', we try to swith it with the next token
        # if current trans is 'L' ('R'), we try to switch it to 'R' ('L')
        epsilon = 1e-12 # used to eniminate invalid actions
        trans_pos_list = np.argsort([self.transition_probs[i][t] for i, t in enumerate(self.transitions)])
        for trans_pos in trans_pos_list:
            t_prob = self.transition_probs[trans_pos]
            t_ori = self.transitions[trans_pos]
            if t_prob[t_ori] > 1 - epsilon:
                break

            if t_ori == 0: # Left-Arc
                new_transitions = deepcopy(self.transitions)
                new_transitions[trans_pos] = 1
            elif t_ori == 1: # Right-Arc
                new_transitions = deepcopy(self.transitions)
                new_transitions[trans_pos] = 0
            elif t_ori == 2: # Shift
                # skip when both current trans and next trans are 'S'
                if trans_pos == len(self.transitions) - 1 or self.transitions[trans_pos + 1] == 2: 
                    continue
                else: # swith current trans and next trans
                    new_transitions = deepcopy(self.transitions)
                    new_transitions[trans_pos] = new_transitions[trans_pos+1]
                    new_transitions[trans_pos+1] = t_ori
            dependencies = syntax.convert_trans2dep(new_transitions)
            et = AST(self.sentence, dependencies, self.semantics)
            if et.res() is not None and et.res() == y:
                et.joint_prob = joint_prob(self.sentence, new_transitions, self.semantics, self.sent_probs, self.transition_probs)
                return et
        return None

    
class Jointer:
    def __init__(self, config=None):
        super(Jointer, self).__init__()
        self.config = config
        self.perception = perception.build(config)
        self.syntax = syntax.build(config)
        self.semantics = semantics.build(config)
        self.ASTs = []
        self.buffer = []
        self.epoch = 0
        self.learning_schedule = ['semantics'] * (0 if config.semantics else 1) \
                                + ['perception'] * (0 if config.perception else 10) \
                                + ['syntax'] * (0 if config.syntax else 5)

    @property
    def learned_module(self):
        return self.learning_schedule[self.epoch % len(self.learning_schedule)]

    def save(self, save_path, epoch=None):
        model = {'epoch': epoch}
        model['perception'] = self.perception.save()
        model['syntax'] = self.syntax.save()
        model['semantics'] = self.semantics.save()
        torch.save(model, save_path)

    def load(self, load_path):
        model = torch.load(load_path)
        self.perception.load(model['perception'])
        self.syntax.load(model['syntax'])
        self.semantics.load(model['semantics'])
        return model['epoch']

    def print(self):
        if self.config.perception:
            print('use ground-truth perception.')
        else:
            print(self.perception.model)
        if self.config.syntax:
            print('use ground-truth syntax.')
        else:
            print(self.syntax.model)
        if self.config.semantics:
            print('use ground-truth semantics.')
        else:
            self.semantics._print_semantics()

    def train(self):
        self.perception.train()
        self.syntax.train()
        # self.semantics.train()
    
    def eval(self):
        self.perception.eval()
        self.syntax.eval()
        # self.semantics.eval()

    def to(self, device):
        self.perception.to(device)
        self.syntax.to(device)
    
    def deduce(self, sample):
        config = self.config
        img_seq = sample['img_seq']
        lengths = sample['len']
        img_seq = img_seq.to(DEVICE)

        if config.perception: # use gt perception
            sentences = sample['label_seq']
            sent_probs = np.ones(sentences.shape + (len(SYMBOLS) - 1,))
        else:
            sentences, sent_probs = self.perception(img_seq)
            sentences = sentences.detach()
            sent_probs = sent_probs.detach().cpu().numpy()
        sentences = [x[:l] for x, l in zip(sentences, lengths)]
        sent_probs = [x[:l] for x, l in zip(sent_probs, lengths)]

        if config.syntax: # use gt parse
            parses = []
            for s, dep in zip(sentences, sample['head']):
                pt = syntax.PartialParse(s)
                pt.dependencies = dep
                parses.append(pt)
        else:
            parses = self.syntax(sentences)

        semantics = self.semantics()
        
        self.ASTs = []
        for s_prob, pt in zip(sent_probs, parses):
            et = AST(pt.sentence.cpu().numpy(), pt.dependencies, semantics, pt.transitions, s_prob, pt.probs)
            self.ASTs.append(et)
        results = [x.res() for x in self.ASTs]

        sentences = [s.cpu().numpy() for s in sentences]
        dependencies = [pt.dependencies for pt in parses]
        return sentences, dependencies, results
    
    def abduce(self, gt_values, batch_img_paths):
        for et, y, img_paths in zip(self.ASTs, gt_values.numpy(), batch_img_paths):
            new_et = et.abduce(y, self.learned_module)
            if new_et: 
                new_et.img_paths = img_paths
                self.buffer.append(new_et)
    
    def clear_buffer(self):
        self.buffer = []

    def learn(self):
        assert len(self.buffer) > 0
        self.train()
        print("Hit samples: ", len(self.buffer), ' Ave length: ', round(np.mean([len(x.sentence) for x in self.buffer]), 2))
        pred_symbols = sorted(Counter([y for x in self.buffer for y in x.sentence]).items())
        print("Symbols: ", len(pred_symbols), pred_symbols)
        print("Dep: ", sorted(Counter([tuple(ast.dependencies) for ast in self.buffer]).items()))

        import json
        self.buffer = sorted(self.buffer, key=lambda x: -x.joint_prob)
        dataset = []
        for ast in self.buffer:
            sent = [int(x) for x in ast.sentence]
            imgs = [x.split('/')[0] for x in ast.img_paths]
            dep = ast.dependencies
            root_res = int(ast.root_node.res())
            ast_res = ast.res()
            ast_res = ast_res if ast_res is None else int(ast_res)
            prob = round(ast.joint_prob, 1)
            dataset.append((sent, imgs, dep, root_res, ast_res, prob))
        json.dump(dataset, open('outputs/dataset.json', 'w'))

        if self.learned_module == 'perception':
            dataset = [(img, label) for x in self.buffer for img, label in zip(x.img_paths, x.sentence)]
            n_iters = int(100)
            print("Learn perception with %d samples for %d iterations, "%(len(self.buffer), n_iters), end='', flush=True)
            st = time()
            self.perception.learn(dataset, n_iters=n_iters)
            print("take %d sec."%(time()-st))

        elif self.learned_module == 'syntax':
            dataset = [{'word': x.sentence, 'head': x.dependencies} for x in self.buffer]
            n_iters = int(100)
            print("Learn syntax with %d samples for %d iterations, "%(len(self.buffer), n_iters), end='', flush=True)
            st = time()
            self.syntax.learn(dataset, n_iters=n_iters)
            print("take %d sec."%(time()-st))

        elif self.learned_module == 'semantics':
            dataset = [[] for _ in range(len(SYMBOLS) - 1)]
            for ast in self.buffer:
                queue = [ast.root_node]
                while len(queue) > 0:
                    node = queue.pop()
                    queue.extend(node.children)
                    xs = tuple([x.res() for x in node.children])
                    y = int(node.res())
                    dataset[node.symbol].append((xs, y, ast.joint_prob))
            self.semantics.learn(dataset)

        self.clear_buffer()
        self.epoch += 1

if __name__ == '__main__':
    # from utils import SEMANTICS
    # sentences = ['5!-7-4', '1+5!*8', '8*9!+5+1/9/3!*9*5']
    # dependencies = [[1, 2, 4, 2, -1, 4], [1, -1, 3, 4, 1, 4], [1, 4, 3, 1, 6, 4, -1, 8, 10, 8, 13, 12, 10, 15, 13, 6, 15]]
    # for s, dep in zip(sentences, dependencies):
    #     et = AST(s, dep, SEMANTICS)
    #     print(et.res())

    model = Jointer(None)
    from dataset import HINT, HINT_collate
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    dataset = HINT('train')
    data_loader = DataLoader(dataset, batch_size=32,
                         shuffle=True, num_workers=4, collate_fn=HINT_collate)
    model.train()
    for sample in tqdm(data_loader):
        # sample = next(iter(val_loader))
        res = model.deduce(sample['img_seq'], sample['len'])
        model.abduce(sample['res'], sample['img_paths'])
        # print(len([1 for x, y in zip(res, sample['res']) if x is not None and x == y]), len(model.buffer))
        # model.clear_buffer()
        model.learn()
    print(len(model.buffer))
