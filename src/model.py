"""Full model: backbone + wavelet attention + EDL/standard head."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from wavelet_attention import WaveletAttention, SEBlock, WaveletModuleSEOnly


class ICHModel(nn.Module):
    def __init__(self, backbone='resnet50', use_wavelet=False, use_edl=False,
                 num_classes=2, pretrained=True, in_channels=1,
                 wavelet_pos='layer2', se_reduction=8, se_only=False, idwt_legacy=False):
        super().__init__()
        self.use_wavelet = use_wavelet
        self.use_edl = use_edl
        self.num_classes = num_classes
        self.wavelet_pos = wavelet_pos
        self.se_only = se_only
        self.idwt_legacy = idwt_legacy

        if in_channels == 1:
            self.expand = lambda x: x.repeat(1, 3, 1, 1)
        else:
            self.expand = nn.Identity()

        if backbone == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            net = models.resnet50(weights=weights)
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
            self.layer1 = net.layer1  # 256 ch
            self.layer2 = net.layer2  # 512 ch
            self.layer3 = net.layer3  # 1024 ch
            self.layer4 = net.layer4  # 2048 ch
            feat_dim = 2048
            if use_wavelet:
                if wavelet_pos in ('layer2', 'layer2_layer3'):
                    self.wavelet_l2 = (WaveletModuleSEOnly(512, se_reduction // 4)
                                       if se_only else WaveletAttention(512, se_reduction, legacy_idwt=idwt_legacy))
                if wavelet_pos in ('layer3', 'layer2_layer3'):
                    self.wavelet_l3 = (WaveletModuleSEOnly(1024, se_reduction // 4)
                                       if se_only else WaveletAttention(1024, se_reduction, legacy_idwt=idwt_legacy))
        elif backbone == 'efficientnet_b0':
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            net = models.efficientnet_b0(weights=weights)
            self.stem = nn.Sequential(*list(net.features.children())[:3])
            self.layer1 = net.features[3]
            self.layer2 = net.features[4]
            self.layer3 = nn.Sequential(*list(net.features.children())[5:7])
            self.layer4 = net.features[7]
            feat_dim = 1280
            if use_wavelet:
                if wavelet_pos in ('layer2', 'layer2_layer3'):
                    self.wavelet_l2 = WaveletAttention(80, se_reduction)
                if wavelet_pos in ('layer3', 'layer2_layer3'):
                    self.wavelet_l3 = WaveletAttention(320, se_reduction)
        elif backbone == 'densenet121':
            weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
            net = models.densenet121(weights=weights)
            self.stem = nn.Sequential(net.features.conv0, net.features.norm0,
                                      net.features.relu0, net.features.pool0)
            self.layer1 = net.features.denseblock1
            self.layer2 = nn.Sequential(net.features.transition1, net.features.denseblock2)
            self.layer3 = nn.Sequential(net.features.transition2, net.features.denseblock3)
            self.layer4 = nn.Sequential(net.features.transition3, net.features.denseblock4,
                                        net.features.norm5)
            feat_dim = 1024
            if use_wavelet:
                if wavelet_pos in ('layer2', 'layer2_layer3'):
                    self.wavelet_l2 = WaveletAttention(512, se_reduction)
                if wavelet_pos in ('layer3', 'layer2_layer3'):
                    self.wavelet_l3 = WaveletAttention(1024, se_reduction)
        else:
            raise ValueError(f'Unknown backbone: {backbone}')

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(feat_dim, 256)
        self.dropout = nn.Dropout(0.5)
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.expand(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        if self.use_wavelet and self.wavelet_pos in ('layer2', 'layer2_layer3'):
            x = self.wavelet_l2(x)
        x = self.layer3(x)
        if self.use_wavelet and self.wavelet_pos in ('layer3', 'layer2_layer3'):
            x = self.wavelet_l3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        x = self.dropout(F.relu(self.fc(x)))
        out = self.head(x)
        if self.use_edl:
            out = F.softplus(out)
        return out
