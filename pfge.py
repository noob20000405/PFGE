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

import data
import models
import utils

# ===========================
# Utils: metrics (Acc/NLL/ECE)
# ===========================
def _metrics_from_probs(probs: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """Compute Acc / NLL / ECE(15 bins) from class probabilities."""
    preds = probs.argmax(1)
    acc = float((preds == labels).mean())
    # NLL
    p_true = np.clip(probs[np.arange(labels.size), labels], 1e-12, 1.0)
    nll = float(-np.log(p_true).mean())
    # 15-bin ECE
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


# ===========================
# SnapshotBooster (你的方法)
# ===========================
class SnapshotBooster:
    def __init__(
        self,
        build_model_fn: Callable[[], torch.nn.Module],
        bn_loader_train_aug,   # DataLoader（train/aug）用于 BN 重估
        device: torch.device,
        mode: str = "linear",  # "linear" or "logpool"
        steps: int = 150,
        lr: float = 0.3,
        l2: float = 1e-3,      # toward uniform
        eta: float = 1e-2,     # correlation penalty
        crit_fraction: float = 0.5,  # 选取验证集底部 margin 的比例；=0 或 >=1 表示不用子集
        utils_module=None,     # 传入 utils（必须含 bn_update）
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
    ) -> Iterable[Dict[str, torch.Tensor]]:
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
        # 收集标签
        y_list = []
        for _, y in val_loader:
            y_list.append(y.numpy())
        y = np.concatenate(y_list, 0).astype(np.int64)
        N = y.shape[0]

        # 选难样本子集（可选）
        if ref_for_subset is not None and (0.0 < self.crit_fraction < 1.0):
            S_idx = self._select_critical_indices(ref_for_subset, val_loader, self.crit_fraction)
        else:
            S_idx = None

        # 缓存每个快照在 val 上的 probs/log-probs
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
        C = P_list[0].shape[1]
        P = np.stack(P_list, 0).astype(np.float64)        # [K,N,C]
        if self.mode == "logpool":
            L = np.stack(L_list, 0).astype(np.float64)    # [K,N,C]

        # 相关性矩阵 M（true-class prob 行中心化协方差）
        F_tc = np.stack([Pi[np.arange(N), y] for Pi in P_list], 0).astype(np.float64)  # [K,N]
        F_tc -= F_tc.mean(axis=1, keepdims=True)
        M = (F_tc @ F_tc.T) / float(max(1, N))            # [K,K]
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
            w = torch.softmax(phi, dim=0)  # [K]
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
    ) -> Dict[str, float]:
        alpha = np.asarray(alpha, dtype=np.float64)
        mix_accum = None
        labels_all = []

        for sd, w in zip(self._iter_snapshots(snapshots, ckpt_key=ckpt_key, map_fn=map_fn), alpha):
            m = self.build_model()
            m.load_state_dict(sd, strict=True)
            m.to(self.device)
            self.utils.bn_update(self.bn_loader, m)
            m.eval()

            chunks = []
            for x, y in test_loader:
                x = x.to(self.device, non_blocking=True)
                if self.mode == "logpool":
                    chunks.append(F.log_softmax(m(x), dim=1).float().cpu().numpy())
                else:
                    chunks.append(F.softmax(m(x), dim=1).float().cpu().numpy())
                labels_all.append(y.numpy())
            pred = np.concatenate(chunks, 0)  # [N,C]
            contrib = w * pred
            mix_accum = contrib if mix_accum is None else (mix_accum + contrib)
            del m
            torch.cuda.empty_cache()

        labels = np.concatenate(labels_all, 0)
        if self.mode == "logpool":
            probs = torch.softmax(torch.from_numpy(mix_accum), dim=1).numpy().astype(np.float32)
        else:
            probs = mix_accum.astype(np.float32)
        return _metrics_from_probs(probs, labels)


# ===========================
# PFGE main
# ===========================
parser = argparse.ArgumentParser(description='PFGE training')

parser.add_argument('--dir', type=str, default=None, metavar='DIR',
                    help='training directory (default: /tmp/pfge)')

parser.add_argument('--dataset', type=str, default='CIFAR10', metavar='DATASET',
                    help='dataset name (default: CIFAR10)')
parser.add_argument('--use_test', action='store_true', default=True,
                    help='switches between validation and test set (default: validation)')
parser.add_argument('--data_path', type=str, default=None, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num-workers', type=int, default=4, metavar='N',
                    help='number of workers (default: 4)')
parser.add_argument("--split_classes", type=int, default=None)
parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--ckpt', type=str, default=None, metavar='CKPT',
                    help='checkpoint to eval (default: None)')

parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--cycle', type=int, default=4, metavar='N',
                    help='cycle length in epochs (default: 4, must be even)')
parser.add_argument('--P', type=int, default=10, help='model recording period (default: 10)')
parser.add_argument('--lr_max', type=float, default=0.05, metavar='LR1',
                    help='maximum learning rate in cycle (default: 0.05)')
parser.add_argument('--lr_min', type=float, default=0.0005, metavar='LR2',
                    help='minimum learning rate in cycle (default: 0.0005)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=5e-4, metavar='WD',
                    help='weight decay (default: 5e-4)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

# Booster switches
parser.add_argument('--booster', action='store_true', default=False,
                    help='enable function-side alpha learning on PFGE snapshots')
parser.add_argument('--boost_mode', type=str, default='linear', choices=['linear','logpool'])
parser.add_argument('--boost_steps', type=int, default=150)
parser.add_argument('--boost_lr', type=float, default=0.3)
parser.add_argument('--boost_l2', type=float, default=1e-3)
parser.add_argument('--boost_eta', type=float, default=1e-2)
parser.add_argument('--boost_crit_fraction', type=float, default=0.5,
                    help='use bottom-quantile (by margin) as hard subset; set 0/≥1 to disable')
parser.add_argument('--boost_max_K', type=int, default=0,
                    help='only use last K snapshots for booster (0=use all)')

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
args.device = torch.device("cuda") if use_cuda else torch.device("cpu")

assert args.cycle % 2 == 0, 'Cycle length should be even'
#（可选更严谨）建议保证 P 是 cycle 的整数倍（或至少是 cycle//2 的整数倍）
# if args.P % args.cycle != 0 and args.P % (args.cycle // 2) != 0:
#     print('[Warn] Prefer P to be a multiple of cycle or cycle//2 for stable SWA updates.')

os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'pfge.sh'), 'w') as f:   # 修正文件名：pfge.sh
    f.write(' '.join(sys.argv))
    f.write('\n')

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
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
        use_validation=not args.use_test,
        split_classes=args.split_classes,
    )

print("Preparing model")
print(*model_cfg.args)
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.to(args.device)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.wd)

num_model = args.epochs // args.P
model_list = [model]
swa_n = np.zeros(int(num_model))
optimizer_list = [optimizer]
for i in range(int(num_model)):
    model1 = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    model1.to(args.device)
    optimizers = torch.optim.SGD(model1.parameters(), lr=args.lr_max, momentum=args.momentum, weight_decay=args.wd)
    model_list.append(model1)
    optimizer_list.append(optimizers)

criterion = utils.cross_entropy

# --- robust checkpoint loading ---
ckpt = torch.load(args.ckpt, map_location="cpu")
start_epoch = 0

# 1) state_dict
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
else:
    state_dict = ckpt
model.load_state_dict(state_dict, strict=True)

# 2) optimizer state (optional)
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
# --- end robust checkpoint loading ---


ensemble_size = 0
N_test = len(loaders['test'].dataset)
pfge_predictions_sum = np.zeros((N_test, num_classes), dtype=np.float64)
pfge_targets = None  # 固定一次 targets 以避免歧义

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc',
           'pfge_ens_acc', 'pfge_ens_nll', 'pfge_ens_ece', 'time']

for epoch in range(args.epochs):
    time_ep = time.time()
    lr_schedule = utils.cyclic_learning_rate(epoch, args.cycle, args.lr_max, args.lr_min)
    i = epoch // args.P

    if i == 0:
        train_res = utils.train_epochs(loaders['train'], model, criterion, optimizer,
                                       lr_schedule=lr_schedule, cuda=use_cuda)
        test_res = utils.eval(loaders['test'], model, criterion, cuda=use_cuda)
        time_ep = time.time() - time_ep
        pfge_ens_acc, pfge_ens_nll, pfge_ens_ece = None, None, None

        if (epoch % args.cycle + 1) == args.cycle // 2:
            utils.moving_average(model_list[i+1], model, 1.0 / (swa_n[i] + 1))
            swa_n[i] += 1

        if (epoch + 1) % args.P == 0:
            utils.bn_update(loaders["train"], model_list[i+1])
            pfge_res = utils.predict(loaders["test"], model_list[i+1])
            pfge_predictions = pfge_res["predictions"].astype(np.float64)
            if pfge_targets is None:
                pfge_targets = pfge_res["targets"]
            pfge_predictions_sum += pfge_predictions
            ensemble_size += 1
            probs_eq = (pfge_predictions_sum / float(ensemble_size)).astype(np.float32)
            mets = _metrics_from_probs(probs_eq, pfge_targets)
            pfge_ens_acc, pfge_ens_nll, pfge_ens_ece = \
                100.0 * mets['acc'], mets['nll'], mets['ece']
            print(f"[PFGE {ensemble_size}] Eq-avg Acc={mets['acc']:.4f}  NLL={mets['nll']:.4f}  ECE(15)={mets['ece']:.4f}")
            utils.save_checkpoint(args.dir, epoch + 1, name="pfge", state_dict=model_list[i+1].state_dict())

    else:
        train_res = utils.train_epochs(loaders['train'], model_list[i], criterion, optimizer_list[i],
                                       lr_schedule=lr_schedule, cuda=use_cuda)
        test_res = utils.eval(loaders['test'], model_list[i], criterion, cuda=use_cuda)
        time_ep = time.time() - time_ep

        pfge_ens_acc, pfge_ens_nll, pfge_ens_ece = None, None, None

        if (epoch % args.cycle + 1) == args.cycle // 2:
            utils.moving_average(model_list[i+1], model_list[i], 1.0/(swa_n[i] + 1))
            swa_n[i] += 1

        if (epoch + 1) % args.P == 0:
            utils.bn_update(loaders["train"], model_list[i + 1])
            pfge_res = utils.predict(loaders["test"], model_list[i + 1])
            pfge_predictions = pfge_res["predictions"].astype(np.float64)
            if pfge_targets is None:
                pfge_targets = pfge_res["targets"]
            pfge_predictions_sum += pfge_predictions
            ensemble_size += 1
            probs_eq = (pfge_predictions_sum / float(ensemble_size)).astype(np.float32)
            mets = _metrics_from_probs(probs_eq, pfge_targets)
            pfge_ens_acc, pfge_ens_nll, pfge_ens_ece = \
                100.0 * mets['acc'], mets['nll'], mets['ece']
            print(f"[PFGE {ensemble_size}] Eq-avg Acc={mets['acc']:.4f}  NLL={mets['nll']:.4f}  ECE(15)={mets['ece']:.4f}")
            utils.save_checkpoint(args.dir, epoch + 1, name="pfge", state_dict=model_list[i + 1].state_dict())

    values = [epoch + 1, lr_schedule(1.0), train_res['loss'], train_res['accuracy'],
              test_res['loss'], test_res['accuracy'],
              pfge_ens_acc, pfge_ens_nll, pfge_ens_ece, time_ep]
    table = tabulate.tabulate([values], columns, tablefmt='simple',
                              floatfmt='9.4f', missingval='-')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

# =======================
# Booster Stage (optional)
# =======================
if args.booster:
    # 收集 PFGE 快照
    ckpt_paths = sorted(
        glob.glob(os.path.join(args.dir, 'pfge-*.pt')),
        key=lambda p: int(os.path.basename(p).split('-')[1].split('.')[0])
    )
    if args.boost_max_K and args.boost_max_K > 0:
        ckpt_paths = ckpt_paths[-args.boost_max_K:]

    if len(ckpt_paths) == 0:
        print('[Booster] No PFGE snapshots found, skip.')
    else:
        print(f'[Booster] Found {len(ckpt_paths)} snapshots.')

        def _build_model():
            m = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
            m.to(args.device)
            return m

        bn_loader = loaders['train']
        val_loader = loaders['test']

        booster = SnapshotBooster(
            build_model_fn=_build_model,
            bn_loader_train_aug=bn_loader,
            device=args.device,
            mode=args.boost_mode,
            steps=args.boost_steps,
            lr=args.boost_lr,
            l2=args.boost_l2,
            eta=args.boost_eta,
            crit_fraction=args.boost_crit_fraction,
            utils_module=utils,
        )

        # 参考模型用于选子集（用最后一个 SWA 输出）
        ref_sd = torch.load(ckpt_paths[-1], map_location='cpu')['state_dict']
        ref_model = _build_model()
        ref_model.load_state_dict(ref_sd, strict=True)

        # 学 α
        alpha = booster.learn_alpha(
            snapshots=ckpt_paths,
            val_loader=val_loader,
            ref_for_subset=ref_model,
            ckpt_key='state_dict',
        )
        with open(os.path.join(args.dir, 'alpha.json'), 'w') as f:
            json.dump({'alpha': alpha.tolist()}, f, indent=2)

        # 评测（强化后）
        metrics = booster.evaluate(
            snapshots=ckpt_paths,
            test_loader=val_loader,
            alpha=alpha,
            ckpt_key='state_dict',
        )
        print('\n=== PFGE + Booster (function-side) ===')
        print(f"Acc: {metrics['acc']:.4f} | NLL: {metrics['nll']:.4f} | ECE(15): {metrics['ece']:.4f}")

