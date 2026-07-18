import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropRate=0.0):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm1d(input_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(input_dim, output_dim, bias=False)

        self.bn2 = nn.BatchNorm1d(output_dim)
        self.relu2 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(output_dim, output_dim, bias=False)

        self.droprate = dropRate
        self.equalInOut = (input_dim == output_dim)
        self.convShortcut = (not self.equalInOut) and nn.Linear(input_dim, output_dim, bias=False) or None

    def forward(self, x):
        if not self.equalInOut:
            x = self.relu1(self.bn1(x))
            out = self.fc1(x)
        else:
            out = self.relu1(self.bn1(x))
            out = self.fc1(out)
        out = self.relu2(self.bn2(out))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.fc2(out)
        shortcut = x if self.equalInOut else self.convShortcut(x)
        return shortcut + out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, input_dim, output_dim, block, dropRate=0.0):
        super().__init__()
        self.layer = self._make_layer(block, input_dim, output_dim, nb_layers, dropRate)

    def _make_layer(self, block, input_dim, output_dim, nb_layers, dropRate):
        layers = []
        for i in range(nb_layers):
            layers.append(block(input_dim if i == 0 else output_dim, output_dim, dropRate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, num_features, num_classes, depth=10, widen_factor=2, dropRate=0.0):
        super().__init__()
        base = max(32, num_features)
        nUnits = [base, base * widen_factor, base * widen_factor * 2]

        assert (depth - 1) % 3 == 0
        n = (depth - 1) // 3
        block = BasicBlock

        self.fc_in = nn.Linear(num_features, nUnits[0])

        self.block1 = NetworkBlock(n, nUnits[0], nUnits[0], block, dropRate)
        self.block2 = NetworkBlock(n, nUnits[0], nUnits[1], block, dropRate)
        self.block3 = NetworkBlock(n, nUnits[1], nUnits[2], block, dropRate)

        self.bn1 = nn.BatchNorm1d(nUnits[2])
        self.relu = nn.ReLU(inplace=True)
        self.fc_out = nn.Linear(nUnits[2], num_classes)
        self.nUnits = nUnits[2]

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.fc_in(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        return self.fc_out(out)

    def pred_emb(self, x):
        out = self.fc_in(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        emb = self.relu(self.bn1(out))
        return self.fc_out(emb), emb

    def feature_list(self, x):
        out_list = []
        out = self.fc_in(x)
        out = self.block1(out)
        out_list.append(out)
        out = self.block2(out)
        out_list.append(out)
        out = self.block3(out)
        out_list.append(out)
        out = self.relu(self.bn1(out))
        return self.fc_out(out), out_list
