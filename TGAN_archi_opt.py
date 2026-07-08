"""-------------------------------------------------Import libraries-------------------------------------------------"""
import torch
from torch import exp
from torch.nn import ModuleDict, Linear, LeakyReLU, BatchNorm1d, Module, Sigmoid, Sequential, Tanh, Dropout, Softmax, \
    MSELoss, CrossEntropyLoss, BCELoss, ReLU

# Check cuda
cuda = True if torch.cuda.is_available() else False


"""---------------------------------------------------Build Model----------------------------------------------------"""

class Attention(Module):
    def __init__(self, in_features, attention_size):
        super(Attention, self).__init__()
        self.attention_weights = Linear(in_features, attention_size)
        self.attention_score = Linear(attention_size, 1, bias=False)
        self.softmax = Softmax(dim=1)

    def forward(self, x):
        attn_weights = torch.tanh(self.attention_weights(x))
        attn_score = self.attention_score(attn_weights)
        attn_score = self.softmax(attn_score)

        weighted_input = x * attn_score
        return weighted_input

# Latent vector
def reparameterization(mu, logvar, z_dim):
    std = exp(logvar / 2)
    sampled_z = torch.rand(mu.size(0), z_dim).cuda() if cuda else torch.rand(mu.size(0), z_dim)
    z = sampled_z * std + mu
    return z


"""
Generator
------------------
Input: Feature dimension
Output: Latent vector
"""
class Generator(Module):
    def __init__(self, in_out, discrete_features, continuous_features):
        super(Generator, self).__init__()
        self.dim = in_out
        self.h1 = 25
        self.h2 = 20

        seq = [
            Linear(self.dim, self.h1),
            ReLU(),
            # BatchNorm1d(self.h1),
            Dropout(0.2),

            Linear(self.h1, self.h2),
            ReLU(),
            # BatchNorm1d(self.h1),

            Linear(self.h2, self.h2),
            ReLU(),
            # BatchNorm1d(self.h2)
               ]
        self.seq = Sequential(*seq)

        self.discrete_out = {feature: Linear(self.in_out, num_classes)
                             for feature, num_classes in discrete_features.items()}
        self.continuous_out = {feature: Linear(self.in_out, 1)
                               for feature in continuous_features}

        self.discrete_out = ModuleDict(self.discrete_out)
        self.continuous_out = ModuleDict(self.continuous_out)

        self.ce = CrossEntropyLoss()
        self.mse = MSELoss()
        self.softmax = Softmax(dim=-1)
        self.relu = ReLU()

    def forward(self, x):
        shared_features = self.shared(x)

        discrete_outputs = {}
        continuous_outputs = {}

        for feature in self.discrete_features:
            discrete_outputs[feature] = self.discrete_out[feature](shared_features)

        for feature in self.continuous_features:
            continuous_outputs[feature] = self.relu(self.continuous_out[feature](shared_features))

        return discrete_outputs, continuous_outputs



    def compute_loss(self, outputs, targets):
        discrete_outputs, continuous_outputs = outputs
        discrete_targets, continuous_targets = targets
        total_loss = 0
        for feature in self.discrete_features:
            if feature in discrete_targets:
                total_loss += self.ce(discrete_outputs[feature], discrete_targets[feature])
                print("ce", self.ce(discrete_outputs[feature], discrete_targets[feature]) )
        for feature in self.continuous_features:
            if feature in continuous_targets:
                total_loss += self.mse(continuous_outputs[feature], continuous_targets[feature])
                print("mse", self.mse(continuous_outputs[feature], continuous_targets[feature]))
        return total_loss


"""
Discriminator
-------------
Input: Latent vector dim
Output: Binary prediction
"""
class Discriminator(Module):
    def __init__(self, dim):
        super(Discriminator, self).__init__()
        self.h1 = 10
        self.h2 = 5

        seq = [
            Linear(dim, self.h1),
            ReLU(),
            # BatchNorm1d(self.h1),
            # Dropout(0.2),

            Linear(self.h1, self.h1),
            ReLU(),
            # BatchNorm1d(self.h1),

            Linear(self.h1, self.h2),
            ReLU(),
            # BatchNorm1d(self.h2),
            # Dropout(0.1),

            Attention(self.h2, 5),
            ReLU(),

            Linear(self.h2, self.h2),
            ReLU(),
            # BatchNorm1d(self.h2),

            Linear(self.h2, 1),
            Sigmoid()]
        self.seq = Sequential(*seq)
    def forward(self, x):
        x = self.seq(x)
        return x

