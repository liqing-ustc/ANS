from utils import DEVICE, SYMBOLS, ID2SYM, SYM2ID, MISSING_VALUE
import time
from tqdm import tqdm, trange
from collections import Counter, OrderedDict

from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

from dataset import HINT, HINT_collate
from jointer import Jointer

import torch
import numpy as np
import random
torch.multiprocessing.set_sharing_strategy('file_system')

import wandb
import argparse
import sys
import os

def parse_args():
    parser = argparse.ArgumentParser('Give Me A HINT')
    parser.add_argument('--wandb', type=str, default='ANS', help='the project name for wandb.')
    parser.add_argument('--resume', type=str, default=None, help='Resumes training from checkpoint.')
    parser.add_argument('--perception_pretrain', type=str, help='initialize the perception from pretrained models.',
                        default='perception/pretrained_model/model_78.2.pth.tar')
    parser.add_argument('--output_dir', type=str, default='outputs/', help='output directory for storing checkpoints')
    parser.add_argument('--save_model', default='1', choices=['0', '1'])
    parser.add_argument('--seed', type=int, default=0, help="Random seed.")

    parser.add_argument('--train_size', type=float, default=None, help="what perceptage of train data is used.")
    parser.add_argument('--max_op_train', type=int, default=None, help="The maximum number of ops in train.")
    parser.add_argument('--main_dataset_ratio', type=float, default=0, 
            help="The percentage of data from the main training set to avoid forgetting in few-shot learning.")
    parser.add_argument('--fewshot', default=None, choices=list('xyabcd'), help='fewshot concept.')

    parser.add_argument('--perception', default='0', choices=['0', '1'], help='whether to provide perfect perception, i.e., no need to learn')
    parser.add_argument('--syntax', default='0', choices=['0', '1'], help='whether to provide perfect syntax, i.e., no need to learn')
    parser.add_argument('--semantics', default='0', choices=['0', '1'], help='whether to provide perfect semantics, i.e., no need to learn')
    parser.add_argument('--curriculum', default='1', choices=['0', '1'], help='whether to use the pre-defined curriculum')
    parser.add_argument('--Y_combinator', default='1', choices=['0', '1'], help='whether to use the recursion primitive (Y-combinator) in dreamcoder')

    parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
    parser.add_argument('--epochs_eval', type=int, default=10, help='how many epochs per evaluation')

    args = parser.parse_args()
    args.save_model = args.save_model == '1'
    args.curriculum = args.curriculum == '1'
    args.perception = args.perception == '1'
    args.syntax = args.syntax == '1'
    args.semantics = args.semantics == '1'
    args.Y_combinator = args.Y_combinator == '1'
    return args

from nltk.tree import Tree
def draw_parse(sentence, head):
    def build_tree(pos):
        children = [i for i, h in enumerate(head) if h == pos]
        return Tree(sentence[pos], [build_tree(x) for x in children])
    
    root = head.index(-1)
    tree = build_tree(root)
    return tree

def evaluate(model, dataloader, n_steps=1, log_prefix='val'):
    model.eval() 
    res_all = []
    res_pred_all = []
    
    expr_all = []
    expr_pred_all = []

    dep_all = []
    dep_pred_all = []

    metrics = OrderedDict()

    with torch.no_grad():
        for sample in tqdm(dataloader):
            res = sample['res']
            expr = sample['expr']
            dep = sample['head']

            res_preds, expr_preds, dep_preds = model.deduce(sample, n_steps=n_steps)
            
            res_pred_all.append(res_preds)
            res_all.append(res)
            expr_pred_all.extend(expr_preds)
            expr_all.extend(expr)
            dep_pred_all.extend(dep_preds)
            dep_all.extend(dep)

    res_pred_all = np.concatenate(res_pred_all, axis=0)
    res_all = np.concatenate(res_all, axis=0)
    result_acc = (res_pred_all == res_all).mean()
    print("Percentage of missing result: %.2f"%(np.mean(res_pred_all == MISSING_VALUE) * 100))
    

    pred = [y for x in expr_pred_all for y in x]
    gt = [SYM2ID(y) for x in expr_all for y in x]
    mask = np.array([0 if x == SYM2ID('(') or x == SYM2ID(')')  else 1 for x in gt], dtype=bool)
    assert len(gt) == len(pred)
    perception_acc = np.mean([x == y for x,y in zip(pred, gt)])

    report = classification_report(gt, pred, target_names=SYMBOLS)
    cmtx = confusion_matrix(gt, pred, normalize='all')
    cmtx = pd.DataFrame(
        (10000*cmtx).astype('int'),
        index=SYMBOLS,
        columns=SYMBOLS
    )
    print(report)
    print(cmtx)

    pred = [y for x in dep_pred_all for y in x]
    gt = [y for x in dep_all for y in x]
    head_acc = np.mean(np.array(pred)[mask] == np.array(gt)[mask])

    tracked_attrs = ['length', 'symbol', 'digit', 'result', 'eval', 'tree_depth', 'ps_depth']
    for attr in tracked_attrs:
        # print(f"result accuracy by {attr}:")
        attr2ids = getattr(dataloader.dataset, f'{attr}2ids')
        for k, ids in sorted(attr2ids.items()):
            res = res_all[ids]
            res_pred = res_pred_all[ids]
            res_acc = (res == res_pred).mean() if ids else 0.
            k = 'div' if k == '/' else k
            metrics[f'result_acc/{attr}/{k}'] = res_acc
            # print(k, "(%2d%%)"%(100*len(ids)//len(dataloader.dataset)), "%5.2f"%(100 * res_acc))

    metrics['result_acc/avg'] = result_acc
    metrics['perception_acc/avg'] = perception_acc
    metrics['head_acc/avg'] = head_acc
    wandb.log({f'{log_prefix}/{k}': v for k, v in metrics.items()})
    
    print("error cases:")
    errors = np.arange(len(res_all))[res_all != res_pred_all]
    for i in errors[:20]:
        expr_pred = ''.join(map(ID2SYM, expr_pred_all[i]))
        print(expr_all[i], expr_pred, dep_all[i], dep_pred_all[i], res_all[i], res_pred_all[i])
        # tree = draw_parse(expr_pred, dep_pred_all[i])
        # tree.draw()


    return perception_acc, head_acc, result_acc

def train(model, args, st_epoch=0):
    best_acc = 0.0
    batch_size = 32
    train_dataloader = torch.utils.data.DataLoader(args.train_set, batch_size=batch_size,
                         shuffle=True, num_workers=4, collate_fn=HINT_collate)
    eval_dataloader = torch.utils.data.DataLoader(args.val_set, batch_size=batch_size,
                         shuffle=False, num_workers=4, collate_fn=HINT_collate)
    
    max_len = float("inf")
    if args.curriculum:
        curriculum_strategy = dict([
            # (0, 7)
            (0, 3),
            (20, 7),
            (40, 11),
            (60, 15),
            (80, float('inf')),
        ])
        print("Curriculum:", sorted(curriculum_strategy.items()))
        for e, l in sorted(curriculum_strategy.items(), reverse=True):
            if st_epoch >= e:
                max_len = l
                break
        train_set.filter_by_len(max_len=max_len)
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                            shuffle=True, num_workers=4, collate_fn=HINT_collate)
    
    ##########evaluate init model###########
    perception_acc, head_acc, result_acc = evaluate(model, eval_dataloader)
    print('Iter {}: {} (Perception Acc={:.2f}, Head Acc={:.2f}, Result Acc={:.2f})'.format(0, 'val', 100*perception_acc, 100*head_acc, 100*result_acc))
    ########################################

    for epoch in range(st_epoch, args.epochs):
        if args.curriculum and epoch in curriculum_strategy:
            max_len = curriculum_strategy[epoch]
            train_set.filter_by_len(max_len=max_len)
            train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                shuffle=True, num_workers=4, collate_fn=HINT_collate)
            if len(train_dataloader) == 0:
                continue

        since = time.time()
        print('-' * 30)
        print('Epoch {}/{} (max_len={}, data={})'.format(epoch, args.epochs - 1, max_len, len(train_set)))

        for _ in range(len(model.learning_schedule)):
            with torch.no_grad():
                model.train()
                train_result_acc = []
                train_perception_acc = []
                train_head_acc = []
                n_samples = 0
                for sample in tqdm(train_dataloader):
                    res = sample['res'].numpy()
                    res_pred, sent_pred, head_pred = model.deduce(sample)
                    model.abduce(res, sample['img_paths'])
                    acc = np.mean(np.array(res_pred) == res)
                    train_result_acc.append(acc)

                    sent_pred = [y for x in sent_pred for y in x]
                    sent = [y for x in sample['sentence'] for y in x]
                    acc = np.mean(np.array(sent_pred) == np.array(sent))
                    train_perception_acc.append(acc)

                    head_pred = [y for x in head_pred for y in x]
                    head = [y for x in sample['head'] for y in x]
                    mask = np.array([0 if x == SYM2ID('(') or x == SYM2ID(')')  else 1 for x in sent], dtype=bool)
                    acc = np.mean(np.array(head_pred)[mask] == np.array(head)[mask])
                    train_head_acc.append(acc)

                    n_samples += res.shape[0]
                    if len(model.buffer) > 1e4:
                        # get enough examples to learn
                        break

                train_result_acc = np.mean(train_result_acc)
                train_perception_acc = np.mean(train_perception_acc)
                train_head_acc = np.mean(train_head_acc)
                abduce_acc = len(model.buffer) / n_samples
            
            wandb.log({'train/result_acc': train_result_acc, 
                       'train/perception_acc': train_perception_acc, 
                       'train/head_acc': train_head_acc, 
                        f'train/abduce_acc/{model.learned_module}': abduce_acc})
            
            model.learn()
            model.epoch += 1
            
        if ((epoch+1) % args.epochs_eval == 0) or (epoch+1 == args.epochs):
            perception_acc, head_acc, result_acc = evaluate(model, eval_dataloader)
            print('{} (Perception Acc={:.2f}, Head Acc={:.2f}, Result Acc={:.2f})'.format('val', 100*perception_acc, 100*head_acc, 100*result_acc))
            if result_acc > best_acc:
                best_acc = result_acc

            if args.save_model:
                model_path = os.path.join(args.ckpt_dir, "model_%03d.p"%(epoch + 1))
                model.save(model_path, epoch=epoch+1)
                
        time_elapsed = time.time() - since
        print('Epoch time: {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    n_steps = 1
    perception_acc, head_acc, result_acc = evaluate(model, eval_dataloader, n_steps)
    print('{} (Perception Acc={:.2f}, Head Acc={:.2f}, Result Acc={:.2f})'.format('val', 100*perception_acc, 100*head_acc, 100*result_acc))

    # Test
    print('-' * 30)
    print('Evaluate on test set...')
    eval_dataloader = torch.utils.data.DataLoader(args.test_set, batch_size=batch_size,
                         shuffle=False, num_workers=4, collate_fn=HINT_collate)
    perception_acc, head_acc, result_acc = evaluate(model, eval_dataloader, n_steps, log_prefix='test')
    print('{} (Perception Acc={:.2f}, Head Acc={:.2f}, Result Acc={:.2f})'.format('test', 100*perception_acc, 100*head_acc, 100*result_acc))

    print('Final model:')
    model.print()
    return



if __name__ == "__main__":
    args = parse_args()
    sys.argv = sys.argv[:1]
    wandb.init(project=args.wandb, dir=args.output_dir, config=vars(args))
    ckpt_dir = os.path.join(wandb.run.dir, '../ckpt')
    os.makedirs(ckpt_dir)
    args.ckpt_dir = ckpt_dir
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    model = Jointer(args)
    model.to(DEVICE)

    if args.fewshot:
        pretrained = 'bak/model_100.p'
        model.load(pretrained)
    
        train_set = HINT('train')
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=32,
                            shuffle=True, num_workers=4, collate_fn=HINT_collate)
        model.eval() 
        model.buffer_augment = []
        with torch.no_grad():
            for sample in tqdm(train_dataloader):
                model.deduce(sample)
                model.buffer_augment.extend([ast for ast, y in zip(model.ASTs, sample['res']) if ast.res() == y])
                for et, y, img_paths in zip(model.ASTs, sample['res'].numpy(), sample['img_paths']):
                    if et.res() == y:
                        et.img_paths = img_paths
                        model.buffer_augment.append(et)
            print("Number of augment examples: ", len(model.buffer_augment))

        fewshot_concepts = list('abcde')
        concept = fewshot_concepts[args.fewshot]
        SYMBOLS.append(concept)
        model.to('cpu')
        model.extend()
        model.to(DEVICE)

    args.input = 'symbol' if args.perception else 'image'
    train_set = HINT('train', input=args.input, fewshot=args.fewshot, 
                    n_sample=args.train_size, max_op=args.max_op_train,
                    main_dataset_ratio=args.main_dataset_ratio)
    val_set = HINT('val', input=args.input, fewshot=args.fewshot)
    test_set = HINT('test', input=args.input, fewshot=args.fewshot)
    print('train:', len(train_set), 'val:', len(val_set), 'test:', len(test_set))

    if not args.fewshot and args.perception_pretrain and not args.perception:
        model.perception.load({'model': torch.load(args.perception_pretrain)}, image_encoder_only=True)
        model.perception.selflabel(train_set.all_exprs())

    st_epoch = 0
    if args.resume:
        st_epoch = model.load(args.resume)
        if st_epoch is None:
            st_epoch = 0


    print(args)
    model.print()
    wandb.log({'train_examples': len(train_set)})

    args.train_set = train_set
    args.val_set = val_set
    args.test_set = test_set

    train(model, args, st_epoch=st_epoch)

