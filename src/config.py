import argparse

def get_config():
    p = argparse.ArgumentParser(description='Wavelet-EDL ICH Detection')
    # data
    p.add_argument('--data_dir', type=str, default='/root/autodl-tmp/ich_project/data')
    p.add_argument('--img_dir', type=str, default='/root/autodl-tmp/ich_project/data/images')
    p.add_argument('--labels_csv', type=str, default='/root/autodl-tmp/ich_project/data/rsna_slice_labels.csv')
    p.add_argument('--rescale_csv', type=str, default='/root/autodl-tmp/ich_project/data/rsna_rescale_values.csv')
    p.add_argument('--manifest_csv', type=str, default='/root/autodl-tmp/ich_project/data/manifest.csv')
    p.add_argument('--output_dir', type=str, default='/root/autodl-tmp/ich_project/runs')
    # model
    p.add_argument('--backbone', type=str, default='resnet50', choices=['resnet50', 'efficientnet_b0', 'densenet121'])
    p.add_argument('--use_wavelet', action='store_true', default=False)
    p.add_argument('--use_edl', action='store_true', default=False)
    p.add_argument('--num_classes', type=int, default=2)
    p.add_argument('--img_size', type=int, default=224)
    p.add_argument('--pretrained', action='store_true', default=True)
    p.add_argument('--se_reduction', type=int, default=8, help='SE block reduction ratio for wavelet attention')
    p.add_argument('--se_only', action='store_true', default=False,
                   help='parameter-matched SE-only attention (no DWT decomposition)')
    p.add_argument('--wavelet_pos', type=str, default='layer2', choices=['layer2', 'layer3', 'layer2_layer3'])
    # training
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--edl_coeff', type=float, default=0.05, help='KL annealing coefficient for EDL')
    p.add_argument('--edl_loss', type=str, default='log', choices=['mse', 'ce', 'log'])
    p.add_argument('--class_weights', action='store_true', default=False)
    p.add_argument('--patience', type=int, default=10, help='Early stopping patience (0=disabled)')
    # data
    p.add_argument('--wl', type=int, default=40, help='Window level (HU)')
    p.add_argument('--ww', type=int, default=80, help='Window width (HU)')
    p.add_argument('--num_workers', type=int, default=8)
    # eval
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--exp_name', type=str, default='baseline')
    return p.parse_args()
