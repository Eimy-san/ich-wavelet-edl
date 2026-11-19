"""Training script for wavelet-enhanced EDL ICH classification."""
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from config import get_config
from dataset import ICHDataset
from model import ICHModel
from edl_head import get_edl_loss, edl_predict, evidence_to_alpha


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_sampler(df, labels):
    class_counts = np.bincount(labels)
    weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(weights, len(labels), replacement=True)


def train_one_epoch(model, loader, optimizer, device, use_edl, edl_loss_fn,
                    K, epoch, total_epochs, edl_coeff, class_weights):
    model.train()
    total_loss = 0
    all_probs, all_labels = [], []

    for batch_idx, (imgs, labels, _) in enumerate(loader):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_edl:
            evidence = model(imgs)
            target = F.one_hot(labels, K).float()
            anneal = min(1.0, epoch / (total_epochs * 0.5)) * edl_coeff
            loss = edl_loss_fn(evidence, target, K, anneal)
            alpha, S = evidence_to_alpha(evidence)
            probs = (alpha / S)[:, 1].detach()
        else:
            logits = model(imgs)
            if class_weights is not None:
                cw = torch.tensor(class_weights, device=device, dtype=torch.float32)
                loss = F.cross_entropy(logits, labels, weight=cw)
            else:
                loss = F.cross_entropy(logits, labels)
            probs = F.softmax(logits, dim=1)[:, 1].detach()

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        all_probs.append(probs.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0
    preds = (all_probs > 0.5).astype(int)
    f1 = f1_score(all_labels, preds)
    acc = accuracy_score(all_labels, preds)
    return total_loss / len(loader), auc, f1, acc


@torch.no_grad()
def evaluate(model, loader, device, use_edl, K):
    model.eval()
    all_probs, all_labels, all_unc = [], [], []

    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        if use_edl:
            evidence = model(imgs)
            probs, unc = edl_predict(evidence)
            all_probs.append(probs[:, 1].cpu().numpy())
            all_unc.append(unc.cpu().numpy())
        else:
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            all_probs.append(probs[:, 1].cpu().numpy())
            all_unc.append(entropy.cpu().numpy())
        all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)
    all_unc = np.concatenate(all_unc)

    auc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0
    preds = (all_probs > 0.5).astype(int)
    f1 = f1_score(all_labels, preds)
    acc = accuracy_score(all_labels, preds)
    tp = ((preds == 1) & (all_labels == 1)).sum()
    fn = ((preds == 0) & (all_labels == 1)).sum()
    fp = ((preds == 1) & (all_labels == 0)).sum()
    tn = ((preds == 0) & (all_labels == 0)).sum()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    return {
        'auc': auc, 'f1': f1, 'acc': acc, 'sens': sens, 'spec': spec,
        'probs': all_probs, 'labels': all_labels, 'unc': all_unc,
    }


def main():
    cfg = get_config()
    set_seed(cfg.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print(f'Exp: {cfg.exp_name} | backbone={cfg.backbone} '
          f'wavelet={cfg.use_wavelet} edl={cfg.use_edl}')

    split_dir = os.path.join(cfg.data_dir, 'splits')
    train_ds = ICHDataset(os.path.join(split_dir, 'train.csv'),
                          cfg.rescale_csv, cfg.img_size, cfg.wl, cfg.ww,
                          augment=True, channels=1)
    val_ds = ICHDataset(os.path.join(split_dir, 'val.csv'),
                        cfg.rescale_csv, cfg.img_size, cfg.wl, cfg.ww,
                        augment=False, channels=1)
    test_ds = ICHDataset(os.path.join(split_dir, 'test.csv'),
                         cfg.rescale_csv, cfg.img_size, cfg.wl, cfg.ww,
                         augment=False, channels=1)

    train_labels = train_ds.df['any'].values
    sampler = make_sampler(train_ds.df, train_labels)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, sampler=sampler,
                              num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)
    print(f'Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')

    model = ICHModel(cfg.backbone, cfg.use_wavelet, cfg.use_edl,
                     cfg.num_classes, cfg.pretrained, in_channels=1,
                     wavelet_pos=cfg.wavelet_pos, se_reduction=cfg.se_reduction,
                     se_only=cfg.se_only, idwt_legacy=cfg.idwt_legacy).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

    if cfg.use_edl:
        edl_loss_fn = get_edl_loss(cfg.edl_loss)
    else:
        edl_loss_fn = None

    class_weights = None
    if cfg.class_weights:
        counts = np.bincount(train_labels)
        class_weights = [len(train_labels) / (2 * c) for c in counts]

    K = cfg.num_classes
    exp_dir = os.path.join(cfg.output_dir, cfg.exp_name + ("_s%d" % cfg.seed if "_s" not in cfg.exp_name else ""))
    os.makedirs(exp_dir, exist_ok=True)
    best_auc = 0
    history = []
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_loss, train_auc, train_f1, train_acc = train_one_epoch(
            model, train_loader, optimizer, device, cfg.use_edl, edl_loss_fn,
            K, epoch, cfg.epochs, cfg.edl_coeff, class_weights)
        val_metrics = evaluate(model, val_loader, device, cfg.use_edl, K)
        scheduler.step()
        dt = time.time() - t0

        print(f'Epoch {epoch:3d}/{cfg.epochs} | loss={train_loss:.4f} '
              f'train_auc={train_auc:.4f} val_auc={val_metrics["auc"]:.4f} '
              f'val_f1={val_metrics["f1"]:.4f} val_acc={val_metrics["acc"]:.4f} '
              f'({dt:.1f}s)')

        history.append({
            'epoch': epoch, 'loss': train_loss, 'train_auc': train_auc,
            'val_auc': val_metrics['auc'], 'val_f1': val_metrics['f1'],
            'val_acc': val_metrics['acc'],
        })

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            patience_counter = 0
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'val_auc': best_auc,
                'cfg': vars(cfg),
            }, os.path.join(exp_dir, 'best_model.pt'))
            print(f'  -> Saved best model (val_auc={best_auc:.4f})')
        else:
            patience_counter += 1
            if cfg.patience > 0 and patience_counter >= cfg.patience:
                print(f'  -> Early stopping at epoch {epoch} (patience={cfg.patience})')
                break

    # Final test evaluation
    ckpt = torch.load(os.path.join(exp_dir, 'best_model.pt'), weights_only=False)
    model.load_state_dict(ckpt['model'])
    test_metrics = evaluate(model, test_loader, device, cfg.use_edl, K)
    print(f'\nTest: AUC={test_metrics["auc"]:.4f} F1={test_metrics["f1"]:.4f} '
          f'Acc={test_metrics["acc"]:.4f} Sens={test_metrics["sens"]:.4f} '
          f'Spec={test_metrics["spec"]:.4f}')

    results = {
        'exp_name': cfg.exp_name, 'seed': cfg.seed, 'backbone': cfg.backbone,
        'use_wavelet': cfg.use_wavelet, 'use_edl': cfg.use_edl,
        'test_auc': test_metrics['auc'], 'test_f1': test_metrics['f1'],
        'test_acc': test_metrics['acc'], 'test_sens': test_metrics['sens'],
        'test_spec': test_metrics['spec'], 'best_val_auc': best_auc,
    }
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(exp_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    np.savez(os.path.join(exp_dir, 'test_predictions.npz'),
             probs=test_metrics['probs'], labels=test_metrics['labels'],
             unc=test_metrics['unc'])
    print(f'Results saved to {exp_dir}')


if __name__ == '__main__':
    main()
