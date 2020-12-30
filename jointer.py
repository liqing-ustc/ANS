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

class AST: # Abstract Syntax Tree
    def __init__(self, pt, semantics):
        self.pt = pt
        self.semantics = semantics

        nodes = [Node(s, semantics[s]) for s in pt.sentence]
        for node, h in zip(nodes, pt.head):
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
        # In future, we can consider to search the execution tree in a top-down manner
        if self.root_node.children_res_valid():
            self._res = y
            self.root_node._res = y
            return self
        return None

    def abduce_perception(self, y):
        # abduce over sentence
        for sent_pos in range(len(self.pt.sentence)):
            for sym in range(len(SYMBOLS) - 1):
                pt = deepcopy(self.pt)
                pt.sentence[sent_pos] = sym
                et = AST(pt, self.semantics)
                if et.res() is not None and et.res() == y:
                    return et
        return None

    def abduce_syntax(self, y):
        # rotate the tree w.r.t the root node
        arcs = self.pt.dependencies
        def get_lc(k):
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] < k])

        def get_rc(k):
            return sorted([arc[1] for arc in arcs if arc[0] == k and arc[1] > k], reverse=True)

        root_idx = self.pt.head.index(-1)
        # left rotate
        for i in get_rc(root_idx):
            pt = deepcopy(self.pt)
            pt.head[root_idx] = i
            pt.head[i] = -1
            ch = get_lc(i) # set the parent the leftmost child of i to root_idx
            if len(ch) > 0:
                pt.head[ch[0]] = root_idx

            et = AST(pt, self.semantics)
            if et.res() is not None and et.res() == y:
                return et
            
        # right rotate
        for i in get_lc(root_idx):
            pt = deepcopy(self.pt)
            pt.head[root_idx] = i
            pt.head[i] = -1
            ch = get_rc(i)
            if len(ch) > 0:
                pt.head[ch[0]] = root_idx

            et = AST(pt, self.semantics)
            if et.res() is not None and et.res() == y:
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
        self.learning_schedule = ['perception'] * (0 if config.perception else 10) \
                               + ['syntax'] * (0 if config.syntax else 5) \
                               + ['semantics'] * (0 if config.semantics else 1)

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
        else:
            sentences = self.perception(img_seq)
            sentences = sentences.detach()
        sentences = [list(x[:l]) for x, l in zip(sentences.cpu().numpy(), lengths)]

        if config.syntax: # use gt parse
            parses = []
            for s, head in zip(sentences, sample['head']):
                pt = syntax.PartialParse(s)
                pt.head = head
                parses.append(pt)
        else:
            parses = self.syntax(sentences)

        semantics = self.semantics()
        
        self.ASTs = [AST(pt, semantics) for pt in parses]
        results = [x.res() for x in self.ASTs]
        head = [pt.head for pt in parses]
        return sentences, head, results
    
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
        print("Hit samples: ", len(self.buffer), ' Ave length: ', round(np.mean([len(x.pt.sentence) for x in self.buffer]), 2))
        pred_symbols = sorted(Counter([y for x in self.buffer for y in x.pt.sentence]).items())
        print("Symbols: ", len(pred_symbols), pred_symbols)
        print("Head: ", sorted(Counter([tuple(ast.pt.head) for ast in self.buffer]).items(), key=lambda x: len(x[0])))

        if self.learned_module == 'perception':
            dataset = [(img, label) for x in self.buffer for img, label in zip(x.img_paths, x.pt.sentence)]
            n_iters = int(100)
            print("Learn perception with %d samples for %d iterations, "%(len(self.buffer), n_iters), end='', flush=True)
            st = time()
            self.perception.learn(dataset, n_iters=n_iters)
            print("take %d sec."%(time()-st))

        elif self.learned_module == 'syntax':
            dataset = [{'word': x.pt.sentence, 'head': x.pt.head} for x in self.buffer]
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
                    dataset[node.symbol].append((xs, y))
            self.semantics.learn(dataset)

        self.clear_buffer()
        self.epoch += 1

if __name__ == '__main__':
    # from utils import SEMANTICS
    # sentences = ['5!-7-4', '1+5!*8', '8*9!+5+1/9/3!*9*5']
    # head = [[1, 2, 4, 2, -1, 4], [1, -1, 3, 4, 1, 4], [1, 4, 3, 1, 6, 4, -1, 8, 10, 8, 13, 12, 10, 15, 13, 6, 15]]
    # for s, dep in zip(sentences, head):
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
