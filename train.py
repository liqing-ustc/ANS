from utils import DEVICE, ID2SYM
import time
from tqdm import tqdm
from collections import Counter

from dataset import HINT, HINT_collate
from jointer import Jointer

import torch
import numpy as np
import pickle
np.random.seed(157)


excludes = ['!']
#train_set = HINT('train', numSamples=500, randomSeed=777)
val_set = HINT('val', exclude_symbols=excludes, max_len=7)
# test_set = HINT('test')
test_set = HINT('val', exclude_symbols=excludes)
train_set = HINT('train', exclude_symbols=excludes)
# train_set = HINT('train', exclude_symbols=['!', '/'], n_sample_zero_res=0.2)
# train_set = HINT('val', exclude_symbols=['!', '*', '/'])
print('train:', len(train_set), 'val:', len(val_set), 'test:', len(test_set))


def evaluate(model, dataloader):
    model.eval() 
    res_all = []
    res_pred_all = []
    
    expr_all = []
    expr_pred_all = []

    dep_all = []
    dep_pred_all = []

    for sample in dataloader:
        img_seq = sample['img_seq']
        res = sample['res']
        seq_len = sample['len']
        expr = sample['expr']
        dep = sample['head']
        img_seq = img_seq.to(DEVICE)

        expr_preds, dep_preds, res_preds = model.deduce(img_seq, seq_len)
        
        res_pred_all.append(res_preds)
        res_all.append(res)
        expr_pred_all.extend(expr_preds)
        expr_all.extend(expr)
        dep_pred_all.extend(dep_preds)
        dep_all.extend(dep)

    res_pred_all = np.concatenate(res_pred_all, axis=0)
    res_all = np.concatenate(res_all, axis=0)
    result_acc = (res_pred_all == res_all).mean()
    
    expr_pred_all = torch.cat(expr_pred_all).cpu().numpy()
    expr_pred_all = ''.join([ID2SYM[x] for x in expr_pred_all])
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

    return perception_acc, syntax_acc, result_acc

def train(model, num_epochs=500, n_epochs_per_eval = 5, st_epoch=0):
    best_acc = 0.0
    reward_moving_average = None
    reward_decay = 0.99
    
    batch_size = 256
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                         shuffle=True, num_workers=4, collate_fn=HINT_collate)
    eval_dataloader = torch.utils.data.DataLoader(val_set, batch_size=128,
                         shuffle=False, num_workers=4, collate_fn=HINT_collate)
    
    max_len = float("inf")
    curriculum_strategy = dict([
        (0, 1),
        (5, 3),
        (50, 5),
        (100, float("inf"))
    ])
    for e, l in curriculum_strategy.items():
        if st_epoch <= e:
            max_len = l
            break
    
    ###########evaluate init model###########
    perception_acc, syntax_acc, result_acc = evaluate(model, eval_dataloader)
    print('{0} (Perception Acc={1:.2f}, Syntax Acc={2:.2f}, Result Acc={3:.2f})'.format('val', 100*perception_acc, 100*syntax_acc, 100*result_acc))
    #########################################

    for epoch in range(st_epoch, num_epochs):
        if epoch in curriculum_strategy:
            max_len = curriculum_strategy[epoch]
            train_set.filter_by_len(max_len=max_len)
            train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                shuffle=True, num_workers=4, collate_fn=HINT_collate)

        since = time.time()
        print('-' * 30)
        print('Epoch {}/{} (max_len={}, data={})'.format(epoch, num_epochs - 1, max_len, len(train_set)))

        # Explore
        with torch.no_grad():
            model.train()
            train_acc = []
            for sample in train_dataloader:
                img_seq = sample['img_seq']
                res = sample['res']
                seq_len = sample['len']
                img_seq = img_seq.to(DEVICE)

                _, _, res_pred = model.deduce(img_seq, seq_len)
                model.abduce(res, sample['img_paths'])
                acc = np.mean(np.array(res_pred) == res.numpy())
                train_acc.append(acc)
            train_acc = np.mean(train_acc)
            abduce_acc = len(model.buffer) / len(train_set)
            print("Train acc: %.2f (abduce %.2f)"%(train_acc * 100, abduce_acc * 100))
        
        print("Dep: ", Counter([tuple(ast.dependencies) for ast in model.buffer]))
        model.learn()
            
        if (epoch+1) % n_epochs_per_eval == 0:
            perception_acc, syntax_acc, result_acc = evaluate(model, eval_dataloader)
            print('{0} (Perception Acc={1:.2f}, Syntax Acc={2:.2f}, Result Acc={3:.2f})'.format('val', 100*perception_acc, 100*syntax_acc, 100*result_acc))
            if result_acc > best_acc:
                best_acc = result_acc

            model_path = "outputs/model_%03d.p"%(epoch + 1)
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
    eval_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                         shuffle=False, num_workers=4, collate_fn=HINT_collate)
    perception_acc, syntax_acc, result_acc = evaluate(model, eval_dataloader)
    print('{0} (Perception Acc={1:.2f}, Syntax Acc={2:.2f}, Result Acc={2:.2f})'.format('test', 100*perception_acc, 100*syntax_acc, 100*result_acc))
    return


model = Jointer()
model.to(DEVICE)
st_epoch = 0
resume = None
resume = "outputs/model_005.p"
if resume:
    st_epoch = model.load(resume)
    if st_epoch is None:
        st_epoch = 0

train(model, st_epoch=st_epoch)

