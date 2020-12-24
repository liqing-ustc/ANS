from utils import DEVICE, SYMBOLS, ID2SYM
import time
from tqdm import tqdm
from collections import Counter

from dataset import HINT, HINT_collate
from jointer import Jointer

import torch
import numpy as np
import random
torch.multiprocessing.set_sharing_strategy('file_system')

import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser('Give Me A HINT')
    parser.add_argument('--excludes', type=str, default='!', help='symbols to be excluded from the dataset')
    parser.add_argument('--resume', type=str, default=None, help='Resumes training from checkpoint.')
    parser.add_argument('--perception-pretrain', type=str, help='initialize the perception from pretrained models.',
                        default='/home/qing/Desktop/Closed-Loop-Learning/perception-pretrain/supervised/perception_60')
    parser.add_argument('--output-dir', type=str, default='outputs/', help='output directory for storing checkpoints')
    parser.add_argument('--seed', type=int, default=314, help="Random seed.")

    parser.add_argument('--perception', action="store_true", help='whether to provide perfect perception, i.e., no need to learn')
    parser.add_argument('--syntax', action="store_true", help='whether to provide perfect syntax, i.e., no need to learn')
    parser.add_argument('--semantics', action="store_true", help='whether to provide perfect semantics, i.e., no need to learn')
    parser.add_argument('--curriculum', action="store_true", help='whether to use the pre-defined curriculum')

    parser.add_argument('--epochs', type=int, default=100, help='number of epochs for training')
    parser.add_argument('--epochs_eval', type=int, default=1, help='how many epochs per evaluation')
    args = parser.parse_args()
    return args

def evaluate(model, dataloader):
    model.eval() 
    res_all = []
    res_pred_all = []
    
    expr_all = []
    expr_pred_all = []

    dep_all = []
    dep_pred_all = []

    for sample in dataloader:
        res = sample['res']
        expr = sample['expr']
        dep = sample['head']

        expr_preds, dep_preds, res_preds = model.deduce(sample)
        
        res_pred_all.append(res_preds)
        res_all.append(res)
        expr_pred_all.extend(expr_preds)
        expr_all.extend(expr)
        dep_pred_all.extend(dep_preds)
        dep_all.extend(dep)

    res_pred_all = np.concatenate(res_pred_all, axis=0)
    res_all = np.concatenate(res_all, axis=0)
    result_acc = (res_pred_all == res_all).mean()
    
    expr_pred_all = np.concatenate(expr_pred_all)
    expr_pred_all = ''.join([ID2SYM(x) for x in expr_pred_all])
    expr_all = ''.join(expr_all)
    assert len(expr_all) == len(expr_pred_all)
    perception_acc = np.mean([x == y for x,y in zip(expr_pred_all, expr_all)])

    dep_all = [y for x in dep_all for y in x]
    dep_pred_all = [y for x in dep_pred_all for y in x]
    syntax_acc = np.mean([x == y for x,y in zip(dep_pred_all, dep_all)])

    print("result accuracy by length:")
    for k in sorted(dataloader.dataset.len2ids.keys()):
        ids = dataloader.dataset.len2ids[k]
        res = res_all[ids]
        res_pred = res_pred_all[ids]
        res_acc = (res == res_pred).mean()
        print(k, "(%2d%%)"%(100*len(ids)//len(dataloader.dataset)), "%5.2f"%(100 * res_acc))
    
    print("result accuracy by symbol:")
    for k in sorted(dataloader.dataset.sym2ids.keys()):
        ids = dataloader.dataset.sym2ids[k]
        res = res_all[ids]
        res_pred = res_pred_all[ids]
        res_acc = (res == res_pred).mean()
        print(k, "(%2d%%)"%(100*len(ids)//len(dataloader.dataset)), "%5.2f"%(100 * res_acc))

    print("result accuracy by digit:")
    for k in sorted(dataloader.dataset.digit2ids.keys()):
        ids = dataloader.dataset.digit2ids[k]
        res = res_all[ids]
        res_pred = res_pred_all[ids]
        res_acc = (res == res_pred).mean()
        print(k, "(%2d%%)"%(100*len(ids)//len(dataloader.dataset)), "%5.2f"%(100 * res_acc))

    print("result accuracy by res:")
    for k in sorted(dataloader.dataset.res2ids.keys())[:10]:
        ids = dataloader.dataset.res2ids[k]
        res = res_all[ids]
        res_pred = res_pred_all[ids]
        res_acc = (res == res_pred).mean()
        print(k, "(%2d%%)"%(100*len(ids)//len(dataloader.dataset)), "%5.2f"%(100 * res_acc))

    print("result accuracy by condition:")
    for k in sorted(dataloader.dataset.cond2ids.keys()):
        ids = dataloader.dataset.cond2ids[k]
        res = res_all[ids]
        res_pred = res_pred_all[ids]
        if len(ids) == 0:
            res_acc = 0.
        else:
            res_acc = (res == res_pred).mean()
        print(k, "(%2d%%)"%(100*len(ids)//len(dataloader.dataset)), "%5.2f"%(100 * res_acc))
    

    return perception_acc, syntax_acc, result_acc

def train(model, args, st_epoch=0):
    best_acc = 0.0
    batch_size = 256
    train_dataloader = torch.utils.data.DataLoader(args.train_set, batch_size=batch_size,
                         shuffle=True, num_workers=4, collate_fn=HINT_collate)
    eval_dataloader = torch.utils.data.DataLoader(args.val_set, batch_size=128,
                         shuffle=False, num_workers=4, collate_fn=HINT_collate)
    
    max_len = float("inf")
    if args.curriculum:
        curriculum_strategy = dict([
            (0, 1),
            (1, 3),
            (10, 5),
            (20, float("inf"))
        ])
        print("Curriculum:", sorted(curriculum_strategy.items()))
        for e, l in sorted(curriculum_strategy.items(), reverse=True):
            if st_epoch >= e:
                max_len = l
                break
        train_set.filter_by_len(max_len=max_len)
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                            shuffle=True, num_workers=4, collate_fn=HINT_collate)
    
    ###########evaluate init model###########
    perception_acc, syntax_acc, result_acc = evaluate(model, eval_dataloader)
    print('{0} (Perception Acc={1:.2f}, Syntax Acc={2:.2f}, Result Acc={3:.2f})'.format('val', 100*perception_acc, 100*syntax_acc, 100*result_acc))
    #########################################

    for epoch in range(st_epoch, args.epochs):
        if args.curriculum and epoch in curriculum_strategy:
            max_len = curriculum_strategy[epoch]
            train_set.filter_by_len(max_len=max_len)
            train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                shuffle=True, num_workers=4, collate_fn=HINT_collate)

        since = time.time()
        print('-' * 30)
        print('Epoch {}/{} (max_len={}, data={})'.format(epoch, args.epochs - 1, max_len, len(train_set)))

        for _ in range(len(model.learning_schedule)):
            with torch.no_grad():
                model.train()
                train_acc = []
                for sample in tqdm(train_dataloader):
                    res = sample['res']
                    _, _, res_pred = model.deduce(sample)
                    model.abduce(res, sample['img_paths'])
                    acc = np.mean(np.array(res_pred) == res.numpy())
                    train_acc.append(acc)
                train_acc = np.mean(train_acc)
                abduce_acc = len(model.buffer) / len(train_set)
                print("Train acc: %.2f (abduce %.2f)"%(train_acc * 100, abduce_acc * 100))
            
            model.learn()
            
        if (epoch+1) % args.epochs_eval == 0:
            perception_acc, syntax_acc, result_acc = evaluate(model, eval_dataloader)
            print('{0} (Perception Acc={1:.2f}, Syntax Acc={2:.2f}, Result Acc={3:.2f})'.format('val', 100*perception_acc, 100*syntax_acc, 100*result_acc))
            if result_acc > best_acc:
                best_acc = result_acc

            model_path = args.output_dir + "model_%03d.p"%(epoch + 1)
            model.save(model_path, epoch=epoch+1)
                
        time_elapsed = time.time() - since
        print('Epoch time: {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

    print('-' * 30)
    perception_acc, syntax_acc, result_acc = evaluate(model, eval_dataloader)
    print('{0} (Perception Acc={1:.2f}, Syntax Acc={2:.2f}, Result Acc={3:.2f})'.format('val', 100*perception_acc, 100*syntax_acc, 100*result_acc))
    if result_acc > best_acc:
        best_acc = result_acc
    print('Best val acc: {:.2f}'.format(100*best_acc))
    # load best model weights
    # model.load_state_dict(best_model_wts)

    # Test
    print('-' * 30)
    print('Evaluate on test set...')
    eval_dataloader = torch.utils.data.DataLoader(args.test_set, batch_size=64,
                         shuffle=False, num_workers=4, collate_fn=HINT_collate)
    perception_acc, syntax_acc, result_acc = evaluate(model, eval_dataloader)
    print('{0} (Perception Acc={1:.2f}, Syntax Acc={2:.2f}, Result Acc={3:.2f})'.format('test', 100*perception_acc, 100*syntax_acc, 100*result_acc))
    return



if __name__ == "__main__":
    args = parse_args()
    sys.argv = sys.argv[:1]

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    excludes = args.excludes
    for sym in excludes:
        SYMBOLS.remove(sym)
    train_set = HINT('train', exclude_symbols=excludes)
    exprs_train = set([x['expr'] for x in train_set])
    val_set = HINT('val', exclude_symbols=excludes, max_len=7, seen_exprs=exprs_train)
    # test_set = HINT('val', exclude_symbols=excludes)
    test_set = HINT('test', exclude_symbols=excludes, seen_exprs=exprs_train)
    print('train:', len(train_set), 'val:', len(val_set), 'test:', len(test_set))

    model = Jointer(args)
    model.to(DEVICE)
    st_epoch = 0
    if args.resume:
        st_epoch = model.load(args.resume)
        if st_epoch is None:
            st_epoch = 0

    if args.perception_pretrain:
        model.perception.load(torch.load(args.perception_pretrain))

    print(args)
    model.print()
    args.train_set = train_set
    args.val_set = val_set
    args.test_set = test_set

    train(model, args, st_epoch=st_epoch)

