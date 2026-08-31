import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1),
                               stride=(stride, 1), padding=(1, 0), bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=(3, 1),
                               stride=(1, 1), padding=(1, 0), bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(
            in_planes, out_planes, kernel_size=(1, 1), stride=(stride, 1),
            padding=0, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        if self.equalInOut:
            out = self.relu2(self.bn2(self.conv1(out)))
        else:
            out = self.relu2(self.bn2(self.conv1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        if not self.equalInOut:
            return torch.add(self.convShortcut(x), out)
        else:
            return torch.add(x, out)


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0):
        super(NetworkBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate)

    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class TemporalWideResNet(nn.Module):
    def __init__(self, depth, num_classes, num_feats, slice_len=None,
                 widen_factor=1, dropRate=0.0, in_channels=1, embed_features=True):
        super(TemporalWideResNet, self).__init__()
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0), "depth must satisfy (depth - 4) % 6 == 0"
        n = (depth - 4) // 6
        block = BasicBlock

        self.embed_features = embed_features
        if embed_features:
            self.conv1 = nn.Conv2d(in_channels, nChannels[0],
                                   kernel_size=(1, num_feats), stride=(1, 1),
                                   padding=0, bias=False)
        else:
            self.conv1 = nn.Conv2d(in_channels, nChannels[0],
                                   kernel_size=(3, 1), stride=(1, 1),
                                   padding=(1, 0), bias=False)


        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 1, dropRate)
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate)
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate)

        self.bn1 = nn.BatchNorm2d(nChannels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        if slice_len is not None:
            min_len_needed = 4
            downsampled = slice_len // 4
            if downsampled < 1:
                raise ValueError(
                    f"slice_len={slice_len} is too short for depth={depth}: "
                    f"two stride-2 blocks would collapse the time axis to < 1. "
                    f"Use a longer slice_len, or reduce the number of "
                    f"stride-2 blocks in this architecture."
                )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n_ = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n_))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _prepare_input(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        return x

    def forward(self, x):
        out = self._prepare_input(x)
        out = self.conv1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.avgpool(out)
        out = out.view(-1, self.nChannels)
        return self.fc(out)

    def pred_emb(self, x):
        out = self._prepare_input(x)
        out = self.conv1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = self.avgpool(out)
        out = out.view(-1, self.nChannels)
        return self.fc(out), out

    def intermediate_forward_simple(self, x, layer_index=None):
        out = self._prepare_input(x)
        out = self.conv1(out)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        return out

    def intermediate_forward(self, x, layer_index=None):
        return self.intermediate_forward_simple(x, layer_index)

    def feature_list(self, x):
        out_list = []
        out = self._prepare_input(x)
        out = self.conv1(out)
        out = self.block1(out)
        out_list.append(out)
        out = self.block2(out)
        out_list.append(out)
        out = self.block3(out)
        out_list.append(out)
        out = self.relu(self.bn1(out))
        out = self.avgpool(out)
        out = out.view(-1, self.nChannels)
        return self.fc(out), out_list

