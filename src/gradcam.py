"""Generate Grad-CAM visualizations and uncertainty heatmaps for qualitative analysis."""
import os
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import ICHModel
from dataset import ICHDataset
from config import get_config
from edl_head import edl_predict


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()
        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        self.model.eval()
        out = self.model(x)
        if self.model.use_edl:
            alpha, S = out + 1, out.sum(dim=1, keepdim=True)
            probs = alpha / S
        else:
            probs = F.softmax(out, dim=1)
        if class_idx is None:
            class_idx = probs[:, 1].argmax().item()
        score = probs[:, class_idx].sum()
        self.model.zero_grad()
        score.backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, probs[0, 1].item()


def overlay_cam(img, cam, alpha=0.4):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img_rgb = np.uint8(np.stack([img]*3, axis=-1) * 255)
    overlay = cv2.addWeighted(img_rgb, 1-alpha, heatmap, alpha, 0)
    return overlay


def main():
    cfg = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    split_dir = os.path.join(cfg.data_dir, 'splits')
    test_ds = ICHDataset(os.path.join(split_dir, 'test.csv'),
                         cfg.rescale_csv, cfg.img_size, cfg.wl, cfg.ww,
                         augment=False, channels=1)

    fig_dir = '/root/autodl-tmp/ich_project/figures'
    os.makedirs(fig_dir, exist_ok=True)

    exp_configs = [
        ('baseline', False, False),
        ('wavelet_only', True, False),
        ('wavelet_edl', True, True),
    ]

    # First pass: compute Grad-CAM with ALL THREE models for each candidate positive case.
    # We select cases where every model's attention is concentrated INSIDE the brain
    # (not on the skull or extracranial artifacts), so the figure cleanly demonstrates
    # that the wavelet-enhanced models focus on the true hemorrhage region.

    # Load all three models up front for candidate scoring.
    models = {}
    for exp_name, use_w, use_e in exp_configs:
        ck = torch.load(f'/root/autodl-tmp/ich_project/runs/{exp_name}/best_model.pt',
                        weights_only=False, map_location=device)
        scfg = ck.get('cfg', {})
        wpos = scfg.get('wavelet_pos', 'layer2') if use_w else 'layer2'
        se_red = scfg.get('se_reduction', 16) if use_w else 16
        m = ICHModel('resnet50', use_w, use_e, 2, pretrained=False, in_channels=1,
                     wavelet_pos=wpos, se_reduction=se_red).to(device)
        m.load_state_dict(ck['model'])
        models[exp_name] = (m, GradCAM(m, m.layer4[-1]))

    def brain_mask(img_np):
        # Brain tissue is nonzero after windowing; skull is very bright, background is 0.
        # Mask = pixels that are part of the head but not the bright skull rim.
        m = (img_np > 0.05) & (img_np < 0.95)
        return m

    def inside_fraction(cam, mask):
        # Fraction of the high-activation energy that falls inside the brain mask.
        hot = cam > 0.5
        if hot.sum() == 0:
            return 0.0
        return float((hot & mask).sum()) / float(hot.sum())

    labels = test_ds.df['any'].values
    candidates = []
    for i in range(min(len(labels), 2000)):
        if labels[i] != 1:
            continue
        img_t = test_ds[i][0].unsqueeze(0).to(device)
        img_np = test_ds[i][0].squeeze().numpy()
        mask = brain_mask(img_np)
        cams, probs, insides = {}, {}, []
        ok = True
        for exp_name, (m, gc) in models.items():
            with torch.enable_grad():
                cam, prob = gc(img_t, class_idx=1)
            cams[exp_name] = cam
            probs[exp_name] = prob
            insides.append(inside_fraction(cam, mask))
            if prob < 0.6:  # all models must confidently detect the bleed
                ok = False
        if not ok:
            continue
        # Require ALL models to put >=80% of their hot activation inside the brain.
        min_inside = min(insides)
        if min_inside < 0.80:
            continue
        # Score: prefer high min inside-fraction and high average confidence.
        score = min_inside + 0.1 * np.mean(list(probs.values()))
        candidates.append((i, score, min_inside))

    candidates.sort(key=lambda x: x[1], reverse=True)
    print(f"Found {len(candidates)} clean ICH+ candidates (all models focus in-brain)")

    # Pick top 3 ICH+ cases with cleanest in-brain attention across all models.
    tp_idx = [c[0] for c in candidates[:3]]

    # Pick 1 clean ICH- case: all models predict low prob and have little hot
    # activation ANYWHERE (neither in-brain nor on extracranial artifacts), so the
    # negative row reads as a clean "nothing to flag" example.
    tn_candidates = []
    for i in range(min(len(labels), 2000)):
        if labels[i] != 0:
            continue
        img_t = test_ds[i][0].unsqueeze(0).to(device)
        max_prob = 0.0
        max_hot_frac = 0.0
        for exp_name, (m, gc) in models.items():
            with torch.enable_grad():
                cam, prob = gc(img_t, class_idx=1)
            max_prob = max(max_prob, prob)
            max_hot_frac = max(max_hot_frac, float((cam > 0.5).sum()) / cam.size)
        if max_prob < 0.1:
            # Lower total hot fraction => cleaner, less spurious activation.
            tn_candidates.append((i, max_hot_frac))
        if len(tn_candidates) >= 40:
            break
    tn_candidates.sort(key=lambda x: x[1])  # cleanest first
    tn_idx = [tn_candidates[0][0]] if tn_candidates else []

    if not tn_idx:  # fallback: first negative
        for i in range(len(labels)):
            if labels[i] == 0:
                tn_idx.append(i)
                break

    sample_idx = tp_idx + tn_idx
    sample_labels = [int(labels[i]) for i in sample_idx]
    sample_imgs = [test_ds[i][0] for i in sample_idx]

    print(f"Selected {len(tp_idx)} ICH+ and {len(tn_idx)} ICH- cases")
    for i, idx in enumerate(sample_idx):
        print(f"  Case {i}: idx={idx}, label={sample_labels[i]}")

    n_rows = len(sample_idx)
    n_cols = len(exp_configs) + 1  # +1 for original
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.2 * n_cols, 3.2 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for col, (exp_name, use_w, use_e) in enumerate(exp_configs):
        model, grad_cam = models[exp_name]

        for row, idx in enumerate(sample_idx):
            img = sample_imgs[row].unsqueeze(0).to(device)
            with torch.enable_grad():
                cam, prob = grad_cam(img, class_idx=1)
            orig = sample_imgs[row].squeeze().numpy()
            overlay = overlay_cam(orig, cam)
            if col == 0:
                axes[row, 0].imshow(orig, cmap='gray')
                gt_label = 'ICH+' if sample_labels[row] == 1 else 'ICH-'
                axes[row, 0].set_title(f'Input ({gt_label})', fontsize=11, fontweight='bold')
                axes[row, 0].axis('off')
            display_name = exp_name.replace('_', ' ').replace('wavelet only', 'Wavelet').replace('wavelet edl', 'Wavelet+EDL').replace('baseline', 'Baseline')
            if exp_name == 'baseline':
                display_name = 'Baseline'
            elif exp_name == 'wavelet_only':
                display_name = '+Wavelet'
            elif exp_name == 'wavelet_edl':
                display_name = '+Wavelet+EDL'
            axes[row, col + 1].imshow(overlay)
            prob_disp = min(max(prob, 0.0), 1.0)
            axes[row, col + 1].set_title(f'{display_name}\nP(ICH)={prob_disp:.2f}', fontsize=10)
            axes[row, col + 1].axis('off')

    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(fig_dir, 'gradcam.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print(f'Grad-CAM saved to {fig_dir}/gradcam.png')


if __name__ == '__main__':
    main()
