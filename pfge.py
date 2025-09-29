import argparse
import numpy as np
import os
import sys
import tabulate
import time
import torch
import torch.nn.functional as F
import glob, json
from typing import List, Union, Callable, Iterable, Dict, Any, Optional

# 保证本目录优先（避免多副本导错）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data
import models
import utils

# ===========================
# Utils: metrics (Acc/NLL/ECE)
# ===========================
def _metrics_from_probs(probs: np.ndarray, labels: np.ndarray):
    preds = probs.argmax(1)
    acc = float((preds == labels).mean())
    p_true = np.clip(probs[np.arange(labels.size), labels], 1e-12, 1.0)
    nll = float(-np.log(p_true).mean())
    bins = np.linspace(0, 1, 16)
    conf = probs.max(1)
    correct = (preds == labels).astype(np.float32)
    ece = 0.0
    for i in range(15):
        lo, hi = bins[i], bins[i+1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return {"acc": acc, "nll": nll, "ece": ece}

def _build_model_for_eval():
    m = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    m.to(args.device)
    return m


# ===========================
# SnapshotBooster
# ===========================
class SnapshotBooster:
    def __init__(
        self,
        build_model_fn: Callable[[], torch.nn.Module],
        bn_loader_train_aug,   # train/aug DataLoader for BN re-estimation
        device: torch.device,
        mode: str = "linear",  # "linear" or "logpool"
        steps: int = 150,
        lr: float = 0.3,
        l2: float = 1e-3,
        eta: float = 1e-2,
        crit_fraction: float = 0.5,
        utils_module=None,
    ):
        assert mode in ("linear", "logpool")
        self.build_model = build_model_fn
        self.bn_loader = bn_loader_train_aug
        self.device = device
        self.mode = mode
        self.steps = steps
        self.lr = lr
        self.l2 = l2
        self.eta = eta
        self.crit_fraction = crit_fraction
        self.utils = utils_module

    @staticmethod
    def _iter_snapshots(
        snapshots: Union[List[Dict[str, torch.Tensor]], List[str]],
        ckpt_key: Optional[str] = None,
        map_fn: Optional[Callable[[Any], Dict[str, torch.Tensor]]] = None,
    ):
        if len(snapshots) == 0:
            return
        first = snapshots[0]
        if isinstance(first, dict):
            for sd in snapshots:
                yield sd
        else:
            for p in snapshots:
                ckpt = torch.load(p, map_location="cpu")
                if map_fn is not None:
                    sd = map_fn(ckpt)
                else:
                    if ckpt_key is not None:
                        sd = ckpt[ckpt_key]
                    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
                        sd = ckpt["state_dict"]
                    else:
                        sd = ckpt
                yield sd

    @torch.no_grad()
    def _select_critical_indices(self, ref_model, val_loader, frac: float):
        if not (0.0 < frac < 1.0):
            return None
        self.utils.bn_update(self.bn_loader, ref_model)
        ref_model.eval()
        logits_all, labels_all = [], []
        for x, y in val_loader:
            x = x.to(self.device, non_blocking=True)
            logits_all.append(ref_model(x).detach().cpu())
            labels_all.append(y.detach().cpu())
        logits = torch.cat(logits_all, 0)
        labels = torch.cat(labels_all, 0)
        true = logits[torch.arange(labels.size(0)), labels]
        tmp = logits.clone()
        tmp[torch.arange(labels.size(0)), labels] = -1e9
        other = tmp.max(1).values
        margins = (true - other).numpy()
        thr = np.quantile(margins, frac)
        return np.nonzero(margins <= thr)[0].astype(np.int64)

    def learn_alpha(
        self,
        snapshots: Union[List[Dict[str, torch.Tensor]], List[str]],
        val_loader,
        ref_for_subset: Optional[torch.nn.Module] = None,
        ckpt_key: Optional[str] = None,
        map_fn: Optional[Callable[[Any], Dict[str, torch.Tensor]]] = None,
    ) -> np.ndarray:
        y_list = []
        for _, y in val_loader:
            y_list.append(y.numpy())
        y = np.concatenate(y_list, 0).astype(np.int64)
        N = y.shape[0]

        if ref_for_subset is not None and (0.0 < self.crit_fraction < 1.0):
            S_idx = self._select_critical_indices(ref_for_subset, val_loader, self.crit_fraction)
        else:
            S_idx = None

        P_list, L_list = [], []
        for sd in self._iter_snapshots(snapshots, ckpt_key=ckpt_key, map_fn=map_fn):
            m = self.build_model()
            m.load_state_dict(sd, strict=True)
            m.to(self.device)
            self.utils.bn_update(self.bn_loader, m)
            m.eval()
            probs_chunks, logp_chunks = [], []
            with torch.no_grad():
                for x, _ in val_loader:
                    x = x.to(self.device, non_blocking=True)
                    out = m(x)
                    if self.mode == "logpool":
                        logp_chunks.append(F.log_softmax(out, dim=1).double().cpu().numpy())
                        probs_chunks.append(F.softmax(out, dim=1).double().cpu().numpy())
                    else:
                        probs_chunks.append(F.softmax(out, dim=1).double().cpu().numpy())
            if self.mode == "logpool":
                L_list.append(np.concatenate(logp_chunks, 0).astype(np.float64))
            P_list.append(np.concatenate(probs_chunks, 0).astype(np.float64))
            del m
            torch.cuda.empty_cache()

        K = len(P_list)
        assert K > 0
        P = np.stack(P_list, 0).astype(np.float64)
        if self.mode == "logpool":
            L = np.stack(L_list, 0).astype(np.float64)

        F_tc = np.stack([Pi[np.arange(N), y] for Pi in P_list], 0).astype(np.float64)
        F_tc -= F_tc.mean(axis=1, keepdims=True)
        M = (F_tc @ F_tc.T) / float(max(1, N))
        tr = np.trace(M)
        if np.isfinite(tr) and tr > 1e-12:
            M *= (K / tr)

        phi = torch.zeros(K, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([phi], lr=self.lr)
        y_t = torch.from_numpy(y).long()
        u_t = torch.ones(K, dtype=torch.float64) / float(K)
        M_t = torch.from_numpy(M).to(torch.float64)
        S_t = torch.from_numpy(S_idx).long() if S_idx is not None else None
        P_t = torch.from_numpy(P).to(torch.float64)
        if self.mode == "logpool":
            L_t = torch.from_numpy(L).to(torch.float64)

        for _ in range(1, self.steps + 1):
            opt.zero_grad()
            w = torch.softmax(phi, dim=0)
            if self.mode == "logpool":
                log_mix = torch.einsum('k,knc->nc', w, L_t)
                if S_t is not None:
                    log_sel = log_mix.index_select(0, S_t)
                    y_sel = y_t.index_select(0, S_t)
                    ce = F.nll_loss(F.log_softmax(log_sel, dim=1).to(torch.float64), y_sel)
                else:
                    ce = F.nll_loss(F.log_softmax(log_mix, dim=1).to(torch.float64), y_t)
            else:
                mix = torch.einsum('k,knc->nc', w, P_t)
                if S_t is not None:
                    mix_sel = mix.index_select(0, S_t)
                    y_sel = y_t.index_select(0, S_t)
                    ce = F.nll_loss(torch.log(mix_sel.clamp_min(1e-12)).to(torch.float64), y_sel)
                else:
                    ce = F.nll_loss(torch.log(mix.clamp_min(1e-12)).to(torch.float64), y_t)

            reg_l2 = self.l2 * torch.sum((w - u_t) * (w - u_t))
            quad = self.eta * torch.sum(w * (M_t @ w))
            loss = ce + reg_l2 + quad
            loss.backward()
            opt.step()

        with torch.no_grad():
            alpha = torch.softmax(phi, dim=0).cpu().numpy().astype(np.float64)
        return alpha

    @torch.no_grad()
    def evaluate(
        self,
        snapshots: Union[List[Dict[str, torch.Tensor]], List[str]],
        test_loader,
        alpha: np.ndarray,
        ckpt_key: Optional[str] = None,
        map_fn: Optional[Callable[[Any], Dict[str, torch.Tensor]]] = None,
    ):
        alpha = np.asarray(alpha, dtype=np.float64)
        mix_accum = None
        labels_list = []
        for _, y in test_loader:
            labels_list.append(y.numpy())
        labels = np.concatenate(labels_list, 0)
        for sd, w in zip(self._iter_snapshots(snapshots, ckpt_key=ckpt_key, map_fn=map_fn), alpha):
            m = self.build_model()
            m.load_state_dict(sd, strict=True)
            m.to(self.device)
            self.utils.bn_update(self.bn_loader, m)
            m.eval()
            chunks = []
            for x, _ in test_loader:
                x = x.to(self.device, non_blocking=True)
                if self.mode == "logpool":
                    chunks.append(F.log_softmax(m(x), dim=1).float().cpu().numpy())
                else:
                    chunks.append(F.softmax(m(x), dim=1).float().cpu().numpy())
            pred = np.concatenate(chunks, 0)
            contrib = w * pred
            mix_accum = contrib if mix_accum is None else (mix_accum + contrib)
            del m
            torch.cuda.empty_cache()
        if self.mode == "logpool":
            probs = torch.softmax(torch.from_numpy(mix_accum), dim=1).numpy().astype(np.float32)
        else:
            probs = mix_accum.astype(np.float32)
        return _metrics_from_probs(probs, labels)

# ===========================
# 辅助：等权集成在任意 loader 上评估（用于最终 test）
# ===========================
@torch.no_grad()
def eval_eq_ensemble_on_loader(ckpt_paths, loader, build_model_fn, bn_loader, device):
    if len(ckpt_paths) == 0:
        return {"acc": 0.0, "nll": 0.0, "ece": 0.0}
    labels_list = []
    for _, y in loader:
        labels_list.append(y.numpy())
    labels = np.concatenate(labels_list, 0)
    probs_sum = None
    for pth in ckpt_paths:
        sd = torch.load(pth, map_location="cpu")["state_dict"]
        m = build_model_fn()
        m.load_state_dict(sd, strict=True)
        m.to(device)
        utils.bn_update(bn_loader, m)
        m.eval()
        chunks = []
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            chunks.append(F.softmax(m(x), dim=1).float().cpu().numpy())
        probs = np.concatenate(chunks, 0)
        probs_sum = probs if probs_sum is None else (probs_sum + probs)
        del m
        torch.cuda.empty_cache()
    probs_eq = (probs_sum / float(len(ckpt_paths))).astype(np.float32)
    return _metrics_from_probs(probs_eq, labels)

# ===========================
# PFGE / FGE main
# ===========================
parser = argparse.ArgumentParser(description='PFGE/FGE training')
parser.add_argument('--dir', type=str, default=None, metavar='DIR')
parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH')
parser.add_argument('--batch_size', type=int, default=128, metavar='N')
parser.add_argument('--num-workers', type=int, default=4, metavar='N')
parser.add_argument("--split_classes", type=int, default=None)
parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL')
parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT')

# eval split
mx = parser.add_mutually_exclusive_group()
mx.add_argument('--use_test', dest='use_test', action='store_true',
                help='evaluate/tune on real test set')
mx.add_argument('--use_val',  dest='use_test', action='store_false',
                help='split a validation set from train and evaluate on it')
parser.set_defaults(use_test=True)
parser.add_argument('--val_size', type=int, default=5000)
parser.add_argument('--eval_test_at_end', action='store_true', default=False)

# algo
parser.add_argument('--algo', type=str, default='pfge', choices=['pfge', 'fge'],
                    help='pfge: moving-average snapshots; fge: single-model snapshots at cycle ends')

# schedule
parser.add_argument('--epochs', type=int, default=20, metavar='N')
parser.add_argument('--cycle', type=int, default=4, metavar='N', help='must be even')
parser.add_argument('--P', type=int, default=10, help='PFGE: recording period; FGE: ignored')
parser.add_argument('--lr_max', type=float, default=0.05, metavar='LR1')
parser.add_argument('--lr_min', type=float, default=0.0005, metavar='LR2')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M')
parser.add_argument('--wd', type=float, default=5e-4, metavar='WD')
parser.add_argument('--seed', type=int, default=1, metavar='S')

# Booster
parser.add_argument('--booster', action='store_true', default=False)
parser.add_argument('--boost_mode', type=str, default=None, choices=['linear','logpool'])
parser.add_argument('--boost_modes', type=str, default='both', choices=['both','linear','logpool'])
parser.add_argument('--boost_steps', type=int, default=150)
parser.add_argument('--boost_lr', type=float, default=0.3)
parser.add_argument('--boost_l2', type=float, default=1e-3)
parser.add_argument('--boost_eta', type=float, default=1e-2)
parser.add_argument('--boost_crit_fraction', type=float, default=0.5)
parser.add_argument('--boost_max_K', type=int, default=0)

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
args.device = torch.device("cuda") if use_cuda else torch.device("cpu")
assert args.cycle % 2 == 0, 'Cycle length should be even'

os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'pfge.sh'), 'w') as f:
    f.write(' '.join(sys.argv)); f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)

print(f"Algo = {args.algo} | Using model {args.model}")
model_cfg = getattr(models, args.model)

print("Loading dataset %s from %s" % (args.dataset, args.data_path))
loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        model_cfg.transform_train,
        model_cfg.transform_test,
        use_validation=not args.use_test,
        val_size=args.val_size,
        split_classes=args.split_classes,
    )
eval_loader = loaders['val'] if ('val' in loaders) else loaders['test']
_eval_name = 'val' if ('val' in loaders) else 'te'

print("Preparing model")
print(*model_cfg.args)
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.to(args.device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.wd)

criterion = utils.cross_entropy

# load init checkpoint
ckpt = torch.load(args.ckpt, map_location="cpu")
state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
model.load_state_dict(state_dict, strict=True)
opt_state = None
if isinstance(ckpt, dict):
    opt_state = ckpt.get("optimizer_state", ckpt.get("optimizer", None))
if opt_state is not None:
    try:
        optimizer.load_state_dict(opt_state)
    except Exception as e:
        print(f"[warn] failed to load optimizer state, start with fresh optimizer: {e}")
else:
    print("[info] no optimizer state in ckpt, start with fresh optimizer")

# ensemble buffers
N_eval = len(eval_loader.dataset)
ens_sum = np.zeros((N_eval, num_classes), dtype=np.float64)
ens_targets = None
ens_size = 0

# table columns
columns = ['ep', 'lr', 'tr_loss', 'tr_acc', f'{_eval_name}_loss', f'{_eval_name}_acc',
           'ens_acc', 'ens_nll', 'ens_ece', 'time']

# ==== PFGE-specific prep ====
num_model = int(args.epochs // args.P) if args.algo == 'pfge' else 0
if args.algo == 'pfge':
    if num_model < 1:
        raise ValueError(f"PFGE requires epochs >= P (got epochs={args.epochs}, P={args.P})")
    model_list = [model]
    optimizer_list = [optimizer]
    for _ in range(num_model):
        m1 = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs).to(args.device)
        opt1 = torch.optim.SGD(m1.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.wd)
        model_list.append(m1); optimizer_list.append(opt1)
    swa_n = np.zeros(num_model, dtype=np.int64)
    print(f"[PFGE init] epochs={args.epochs}, P={args.P}, num_model={num_model}, len(model_list)={len(model_list)}")

snap_prefix = 'pfge' if args.algo == 'pfge' else 'fge'

# ================== training loop ==================
for epoch in range(args.epochs):
    t0 = time.time()
    lr_schedule = utils.cyclic_learning_rate(epoch, args.cycle, args.lr_max, args.lr_min)

    # ---- PFGE branch ----
    if args.algo == 'pfge':
        i = epoch // args.P
        if i == 0:
            train_res = utils.train_epochs(loaders['train'], model, criterion, optimizer,
                                           lr_schedule=lr_schedule, cuda=use_cuda)
            eval_res = utils.eval(eval_loader, model, criterion, cuda=use_cuda)
            # mid-cycle SWA into next bucket
            if (epoch % args.cycle + 1) == args.cycle // 2:
                utils.moving_average(model_list[i+1], model, 1.0 / (swa_n[i] + 1))
                swa_n[i] += 1
            # record snapshot every P epochs
            if (epoch + 1) % args.P == 0:
                utils.bn_update(loaders['train'], model_list[i+1])
                res = utils.predict(eval_loader, model_list[i+1])
                probs = res["predictions"].astype(np.float64)
                if ens_targets is None: ens_targets = res["targets"]
                ens_sum += probs; ens_size += 1
                eq = (ens_sum / float(ens_size)).astype(np.float32)
                mets = _metrics_from_probs(eq, ens_targets)
                print(f"[PFGE {ens_size}] Eq-avg Acc={mets['acc']:.4f}  NLL={mets['nll']:.4f}  ECE(15)={mets['ece']:.4f}")
                utils.save_checkpoint(args.dir, epoch + 1, name=snap_prefix, state_dict=model_list[i+1].state_dict())
        else:
            train_res = utils.train_epochs(loaders['train'], model_list[i], criterion, optimizer_list[i],
                                           lr_schedule=lr_schedule, cuda=use_cuda)
            eval_res = utils.eval(eval_loader, model_list[i], criterion, cuda=use_cuda)
            if (epoch % args.cycle + 1) == args.cycle // 2:
                utils.moving_average(model_list[i+1], model_list[i], 1.0/(swa_n[i] + 1))
                swa_n[i] += 1
            if (epoch + 1) % args.P == 0:
                utils.bn_update(loaders['train'], model_list[i+1])
                res = utils.predict(eval_loader, model_list[i+1])
                probs = res["predictions"].astype(np.float64)
                if ens_targets is None: ens_targets = res["targets"]
                ens_sum += probs; ens_size += 1
                eq = (ens_sum / float(ens_size)).astype(np.float32)
                mets = _metrics_from_probs(eq, ens_targets)
                print(f"[PFGE {ens_size}] Eq-avg Acc={mets['acc']:.4f}  NLL={mets['nll']:.4f}  ECE(15)={mets['ece']:.4f}")
                utils.save_checkpoint(args.dir, epoch + 1, name=snap_prefix, state_dict=model_list[i+1].state_dict())

    # ---- FGE branch ----
    else:
        # 单模型，按循环 LR 训练
        train_res = utils.train_epochs(loaders['train'], model, criterion, optimizer,
                                       lr_schedule=lr_schedule, cuda=use_cuda)
        eval_res = utils.eval(eval_loader, model, criterion, cuda=use_cuda)
        # 在每个 cycle 结束时保存一个快照（最小成本、最常用做法）
        end_of_cycle = ((epoch % args.cycle) == (args.cycle - 1))
        if end_of_cycle:
            utils.bn_update(loaders['train'], model)
            res = utils.predict(eval_loader, model)
            probs = res["predictions"].astype(np.float64)
            if ens_targets is None: ens_targets = res["targets"]
            ens_sum += probs; ens_size += 1
            eq = (ens_sum / float(ens_size)).astype(np.float32)
            mets = _metrics_from_probs(eq, ens_targets)
            print(f"[FGE {ens_size}] Eq-avg Acc={mets['acc']:.4f}  NLL={mets['nll']:.4f}  ECE(15)={mets['ece']:.4f}")
            utils.save_checkpoint(args.dir, epoch + 1, name=snap_prefix, state_dict=model.state_dict())

    t_cost = time.time() - t0
    lr_now = lr_schedule(1.0)
    values = [epoch + 1, lr_now, train_res['loss'], train_res['accuracy'],
              eval_res['loss'], eval_res['accuracy'],
              100.0 * (ens_sum.argmax(1) == ens_targets).mean() if ens_size > 0 else None,
              None, None, t_cost]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f', missingval='-')
    if epoch % 40 == 0:
        table = table.split('\n'); table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

# =======================
# Booster Stage (optional) —— 自动跑 linear + logpool
# =======================
if args.booster:
    ckpt_paths = sorted(
        glob.glob(os.path.join(args.dir, f'{snap_prefix}-*.pt')),
        key=lambda p: int(os.path.basename(p).split('-')[1].split('.')[0])
    )
    if args.boost_max_K and args.boost_max_K > 0:
        ckpt_paths = ckpt_paths[-args.boost_max_K:]
    if len(ckpt_paths) == 0:
        print('[Booster] No snapshots found, skip.')
    else:
        print(f'[Booster] Found {len(ckpt_paths)} snapshots.')
        def _build_model():
            m = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
            m.to(args.device); return m
        bn_loader = loaders['train']
        val_loader = eval_loader

        if args.boost_mode is not None:
            modes_to_run = [args.boost_mode]
        else:
            modes_to_run = ['linear', 'logpool'] if args.boost_modes == 'both' else [args.boost_modes]

        # 参考模型：最后一个快照
        ref_sd = torch.load(ckpt_paths[-1], map_location='cpu')['state_dict']
        ref_model = _build_model(); ref_model.load_state_dict(ref_sd, strict=True)

        for mode in modes_to_run:
            booster = SnapshotBooster(
                build_model_fn=_build_model,
                bn_loader_train_aug=bn_loader,
                device=args.device,
                mode=mode,
                steps=args.boost_steps,
                lr=args.boost_lr,
                l2=args.boost_l2,
                eta=args.boost_eta,
                crit_fraction=args.boost_crit_fraction,
                utils_module=utils,
            )
            alpha = booster.learn_alpha(
                snapshots=ckpt_paths, val_loader=val_loader,
                ref_for_subset=ref_model, ckpt_key='state_dict'
            )
            with open(os.path.join(args.dir, f'alpha.{mode}.json'), 'w') as f:
                json.dump({'alpha': alpha.tolist()}, f, indent=2)

            metrics = booster.evaluate(
                snapshots=ckpt_paths, test_loader=val_loader,
                alpha=alpha, ckpt_key='state_dict'
            )
            print(f'\n=== {args.algo.upper()} + Booster [{mode}] on {("val" if "val" in loaders else "test")} ===')
            print(f"Acc: {metrics['acc']:.4f} | NLL: {metrics['nll']:.4f} | ECE(15): {metrics['ece']:.4f}")

            if args.eval_test_at_end and 'test' in loaders and val_loader is not loaders['test']:
                metrics_test = booster.evaluate(
                    snapshots=ckpt_paths, test_loader=loaders['test'],
                    alpha=alpha, ckpt_key='state_dict'
                )
                print(f'\n=== Final Test ({args.algo.upper()} + Booster [{mode}], no tuning) ===')
                print(f"Acc: {metrics_test['acc']:.4f} | NLL: {metrics_test['nll']:.4f} | ECE(15): {metrics_test['ece']:.4f}")

        # 额外：在 test 上报告一次 Eq-Avg（便于对比）
        if args.eval_test_at_end and 'test' in loaders and val_loader is not loaders['test']:
            eq_metrics_test = eval_eq_ensemble_on_loader(
                ckpt_paths=ckpt_paths, loader=loaders['test'],
                build_model_fn=_build_model, bn_loader=bn_loader, device=args.device
            )
            print(f'\n=== Final Test ({args.algo.upper()} Eq-Avg, no tuning) ===')
            print(f"Acc: {eq_metrics_test['acc']:.4f} | NLL: {eq_metrics_test['nll']:.4f} | ECE(15): {eq_metrics_test['ece']:.4f}")

# === Always report Eq-Avg on TEST at the end (independent of booster/mode) ===
ckpt_paths = sorted(
    glob.glob(os.path.join(args.dir, f"{'pfge' if args.algo=='pfge' else 'fge'}-*.pt")),
    key=lambda p: int(os.path.basename(p).split('-')[1].split('.')[0])
)

if len(ckpt_paths) == 0:
    print("[Final Test Eq-Avg] No snapshots found. Skipping.")
elif 'test' not in loaders:
    print("[Final Test Eq-Avg] No real TEST loader present. Skipping.")
else:
    metrics_test = eval_eq_ensemble_on_loader(
        ckpt_paths=ckpt_paths,
        loader=loaders['test'],
        build_model_fn=_build_model_for_eval,
        bn_loader=loaders['train'],
        device=args.device
    )
    print(f"\n=== Final Test ({args.algo.upper()} Eq-Avg) ===")
    print(f"Acc: {metrics_test['acc']:.4f} | NLL: {metrics_test['nll']:.4f} | ECE(15): {metrics_test['ece']:.4f}")

