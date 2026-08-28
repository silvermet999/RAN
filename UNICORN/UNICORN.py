import torch
import torch.nn as nn


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super().__init__()
        layers = [nn.Linear(in_features, out_features), nn.ReLU(inplace=True)]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ForwConvTabular(nn.Module):
    def __init__(self, input_dim, num_classes, embedding_dim=64):
        super().__init__()
        self.stem = LinearBlock(input_dim, 32)
        self.rep_blocks = nn.Sequential(
            LinearBlock(32, 32),
            LinearBlock(32, 32),
        )
        self.extractor_dropout = nn.Dropout(0.25)
        self.fc256 = LinearBlock(32, 256)
        self.fc128 = LinearBlock(256, embedding_dim, dropout=0.25)
        self.fc64 = LinearBlock(embedding_dim, 64, dropout=0.25)
        self.fc_out = nn.Linear(64, num_classes)
    def forward(self, x):
        x = self.stem(x)
        x = self.rep_blocks(x)
        x = self.extractor_dropout(x)

        x = self.fc256(x)
        embedding = self.fc128(x)
        x = self.fc64(embedding)
        logits = self.fc_out(x)

        return logits, embedding


