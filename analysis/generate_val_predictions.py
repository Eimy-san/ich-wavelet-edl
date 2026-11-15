"""Generate validation-set predictions for all four configs (for operating-point
selection and temperature scaling without touching the test set)."""

import os, sys, json
import numpy as np
import torch

sys.path.insert(0, "/root/autodl-tmp/ich_project/code")
os.chdir("/root/autodl-tmp/ich_project/code")
from config import get_config
from dataset import ICHDataset
from model import ICHModel
from train import evaluate
from torch.utils.data import DataLoader

cfg = get_config()
device = torch.device("cuda")
split_dir = os.path.join(cfg.data_dir, "splits")
val_ds = ICHDataset(
    os.path.join(split_dir, "val.csv"),
    cfg.rescale_csv,
    cfg.img_size,
    cfg.wl,
    cfg.ww,
    augment=False,
    channels=1,
)
val_loader = DataLoader(
    val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=8, pin_memory=True
)
cfgs = {
    "baseline": dict(use_wavelet=False, use_edl=False),
    "wavelet_only": dict(use_wavelet=True, use_edl=False),
    "edl_only": dict(use_wavelet=False, use_edl=True),
    "wavelet_edl": dict(use_wavelet=True, use_edl=True),
}
out = {}
for name, kw in cfgs.items():
    ckpt_path = f"/root/autodl-tmp/ich_project/runs/{name}/best_model.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt["cfg"]
    model = ICHModel(
        saved["backbone"],
        saved["use_wavelet"],
        saved["use_edl"],
        saved["num_classes"],
        pretrained=False,
        in_channels=1,
        wavelet_pos=saved.get("wavelet_pos", "layer2"),
        se_reduction=saved.get("se_reduction", 8),
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    m = evaluate(model, val_loader, device, saved["use_edl"], saved["num_classes"])
    np.savez(
        f"/root/autodl-tmp/ich_project/runs/{name}/val_predictions.npz",
        probs=m["probs"],
        labels=m["labels"],
        unc=m["unc"],
    )
    out[name] = {
        "auc": m["auc"],
        "f1": m["f1"],
        "acc": m["acc"],
        "sens": m["sens"],
        "spec": m["spec"],
    }
    print(name, {k: round(v, 4) for k, v in out[name].items()})
    del model
    torch.cuda.empty_cache()
json.dump(out, open("/root/autodl-tmp/ich_project/val_summary.json", "w"), indent=2)
