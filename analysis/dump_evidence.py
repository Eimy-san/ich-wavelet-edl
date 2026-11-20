"""Dump Dirichlet evidence (alphas) for EDL runs on the val and test splits.

For each run directory passed on the command line, loads best_model.pt,
rebuilds the val/test loaders exactly as training-time evaluation did
(shuffle=False, csv order), runs the model, and saves an evidence npz
(alphas, probs, unc, labels) next to the existing predictions.
Recomputed probabilities are asserted to match any existing
test_predictions.npz / val_predictions.npz to guarantee alignment.
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from config import get_config
from dataset import ICHDataset
from model import ICHModel
from edl_head import evidence_to_alpha, edl_predict


@torch.no_grad()
def run_split(model, loader, device, use_edl, K, ref_npz=None):
    model.eval()
    P, U, Y, A = [], [], [], []
    for imgs, labels, _ in loader:
        imgs = imgs.to(device)
        if use_edl:
            evidence = model(imgs)
            alpha, S = evidence_to_alpha(evidence)
            probs, unc = edl_predict(evidence)
            A.append(alpha.cpu().numpy())
            P.append(probs[:, 1].cpu().numpy())
            U.append(unc.cpu().numpy())
        else:
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            ent = -(probs * torch.log(probs + 1e-8)).sum(dim=1)
            P.append(probs[:, 1].cpu().numpy())
            U.append(ent.cpu().numpy())
        Y.append(labels.numpy())
    probs = np.concatenate(P)
    unc = np.concatenate(U)
    labels = np.concatenate(Y)
    alphas = np.concatenate(A) if A else None
    if ref_npz is not None and os.path.exists(ref_npz):
        ref = np.load(ref_npz)
        assert (ref["labels"] == labels).all(), f"label mismatch vs {ref_npz}"
        dp = np.abs(ref["probs"] - probs).max()
        if dp > 5e-3:
            raise RuntimeError(f"prob mismatch ({dp:.2e}) vs {ref_npz}")
        print(f"    aligned with {os.path.basename(ref_npz)} (max diff {dp:.2e})")
    return probs, unc, labels, alphas


def main(run_dirs):
    legacy = "--legacy" in run_dirs
    run_dirs = [r for r in run_dirs if r != "--legacy"]
    argv_backup = sys.argv
    sys.argv = [sys.argv[0]]
    try:
        cfg0 = get_config()
    finally:
        sys.argv = argv_backup
    split_dir = os.path.join(cfg0.data_dir, "splits")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | legacy_idwt={legacy}")
    for rd in run_dirs:
        rd = os.path.abspath(rd)
        name = os.path.basename(rd)
        ckpt = torch.load(
            os.path.join(rd, "best_model.pt"), map_location=device, weights_only=False
        )
        mcfg = ckpt.get("cfg", {})
        if isinstance(mcfg, dict) and "use_edl" in mcfg:
            use_edl = bool(mcfg["use_edl"])
            use_wavelet = bool(mcfg.get("use_wavelet", False))
            se_only = bool(mcfg.get("se_only", False))
        else:
            raise RuntimeError(f"{name}: checkpoint has no cfg")
        model = ICHModel(
            "resnet50",
            use_wavelet,
            use_edl,
            se_reduction=16,
            se_only=se_only,
            idwt_legacy=legacy,
        ).to(device)
        model.load_state_dict(ckpt["model"])
        K = 2
        for split in ["val", "test"]:
            ds = ICHDataset(
                os.path.join(split_dir, f"{split}.csv"),
                cfg0.rescale_csv,
                224,
                40,
                80,
                augment=False,
                channels=1,
            )
            loader = DataLoader(
                ds, batch_size=128, shuffle=False, num_workers=6, pin_memory=True
            )
            ref = os.path.join(rd, f"{split}_predictions.npz")
            probs, unc, labels, alphas = run_split(
                model, loader, device, use_edl, K, ref
            )
            out = dict(probs=probs, unc=unc, labels=labels)
            if alphas is not None:
                out["alphas"] = alphas
            np.savez(os.path.join(rd, f"{split}_evidence.npz"), **out)
            print(
                f"  {name}/{split}: saved evidence ({len(probs)} slices, "
                f"use_edl={use_edl})"
            )


if __name__ == "__main__":
    main(sys.argv[1:])
