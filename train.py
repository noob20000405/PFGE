import argparse
import os
import sys
import tabulate
import time
import torch
import torch.nn.functional as F
import numpy as np  # for probs-based metrics

import data
import models
import utils


parser = argparse.ArgumentParser(description='SGD training')
parser.add_argument('--dir', type=str, default='/home/PFGE/', metavar='DIR',
                    help='training directory (default: /home/PFGE/)')
parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')
parser.add_argument("--split_classes", type=int, default=None)
parser.add_argument('--model', type=str, default=None, metavar='MODEL', required=True,
                    help='model name (default: None)')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')

# === 互斥：用 test 还是切 val ===
mx = parser.add_mutually_exclusive_group()
mx.add_argument('--use_test', dest='use_test', action='store_true',
                help='evaluate on real test set during training')
mx.add_argument('--use_val',  dest='use_test', action='store_false',
                help='split a validation set from train and evaluate on it')
parser.set_defaults(use_test=True)

# 只有 --use_val 时才生效
parser.add_argument('--val_size', type=int, default=5000,
                    help='when using --use_val, number of images held out from train')
# 可选：收尾在真实 test 上评一次（不参与调参）
parser.add_argument('--eval_test_at_end', action='store_true', default=False,
                    help='also evaluate ONCE on the real test set at the very end')

parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=50, metavar='N',
                    help='save frequency (default: 50)')
parser.add_argument('--lr_init', type=float, default=0.01, metavar='LR',
                    help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, metavar='WD',
                    help='weight decay (default: 1e-4)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
args.device = torch.device("cuda") if use_cuda else torch.device("cpu")

os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

print("Using model %s" % args.model)
model_cfg = getattr(models, args.model)

print("Loading dataset %s from %s" % (args.dataset, args.data_path))
loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        model_cfg.transform_train,
        model_cfg.transform_test,
        use_validation=not args.use_test,   # --use_val => use_validation=True
        val_size=args.val_size,
        split_classes=args.split_classes,
    )

# 评估 loader：优先用 val（若没有就用 test）
eval_loader = loaders['val'] if ('val' in loaders) else loaders['test']
_eval_name = 'val' if ('val' in loaders) else 'test'

print("Preparing model")
print(*model_cfg.args)
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.to(args.device)

# ========== metrics helpers (probs-based NLL/ECE) ==========
@torch.no_grad()
def _eval_probs_metrics(probs_list, labels_list):
    probs = np.concatenate(probs_list, axis=0)
    labels = np.concatenate(labels_list, axis=0)
    acc = float((probs.argmax(1) == labels).mean())
    nll = float(-np.log(np.clip(probs[np.arange(labels.size), labels], 1e-12, 1.0)).mean())
    # 15-bin ECE
    bins = np.linspace(0, 1, 16)
    conf = probs.max(1)
    preds = probs.argmax(1)
    correct = (preds == labels).astype(np.float32)
    ece = 0.0
    for i in range(15):
        lo, hi = bins[i], bins[i+1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return {'acc': acc, 'nll': nll, 'ece': ece}

@torch.no_grad()
def _collect_probs(model, loader, device, non_blocking=False):
    model.eval()
    probs_list, labels_list = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=non_blocking)
        logits = model(x)
        p = torch.softmax(logits, dim=1).cpu().numpy()
        probs_list.append(p)
        labels_list.append(y.cpu().numpy())
    return probs_list, labels_list
# ================================================================

def learning_rate_schedule(epoch):
    t = epoch / args.epochs
    lr_ratio = 0.2
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    return args.lr_init * factor

criterion = utils.cross_entropy
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr_init,
    momentum=args.momentum,
    weight_decay=args.wd
)

start_epoch = 0
if args.resume is not None:
    print("Resume training from %s" % args.resume)
    ckpt = torch.load(args.resume, map_location='cpu')
    start_epoch = ckpt.get("epoch", 0)
    # state_dict
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    # optimizer
    opt_state = None
    if isinstance(ckpt, dict):
        opt_state = ckpt.get("optimizer_state", ckpt.get("optimizer", None))
    if opt_state is not None:
        try:
            optimizer.load_state_dict(opt_state)
        except Exception as e:
            print(f"[warn] failed to load optimizer state, continue fresh: {e}")

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', f'{_eval_name}_loss', f'{_eval_name}_acc', 'time']

# save an initial checkpoint
utils.save_checkpoint(
    args.dir,
    start_epoch - 1,
    state_dict=model.state_dict(),
    optimizer_state=optimizer.state_dict()
)

eval_res = {'loss': None, 'accuracy': None}
for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    lr = learning_rate_schedule(epoch)
    utils.adjust_learning_rate(optimizer, lr)

    train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer, cuda=use_cuda)
    eval_res = utils.eval(eval_loader, model, criterion, cuda=use_cuda)

    if ((epoch + 1) % args.save_freq == 0) or ((epoch + 1) == args.epochs):
        utils.save_checkpoint(
            args.dir, epoch + 1,
            state_dict=model.state_dict(),
            optimizer_state=optimizer.state_dict()
        )


    time_ep = time.time() - time_ep
    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'],
              eval_res['loss'], eval_res['accuracy'], time_ep]

    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
    if epoch % 40 == 1 or epoch == start_epoch:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)



# ===== Final evaluation on the chosen eval split (val if available, else test) =====
probs_list, labels_list = _collect_probs(
    model, eval_loader, device=args.device, non_blocking=use_cuda
)
final_metrics = _eval_probs_metrics(probs_list, labels_list)
print(f'\n=== Final Eval on {_eval_name} (probs-based) ===')
print(f"Acc: {final_metrics['acc']:.4f} | NLL: {final_metrics['nll']:.4f} | ECE(15): {final_metrics['ece']:.4f}")

# ===== (Optional) also evaluate ONCE on the real test set, no tuning =====
if args.eval_test_at_end and 'test' in loaders and _eval_name != 'test':
    probs_list_t, labels_list_t = _collect_probs(
        model, loaders['test'], device=args.device, non_blocking=use_cuda
    )
    test_metrics = _eval_probs_metrics(probs_list_t, labels_list_t)
    print('\n=== Final Test (no tuning) ===')
    print(f"Acc: {test_metrics['acc']:.4f} | NLL: {test_metrics['nll']:.4f} | ECE(15): {test_metrics['ece']:.4f}")

