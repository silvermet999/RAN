import argparse
import sys

import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
import torch
from data import main

cuda = True if torch.cuda.is_available() else False

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--X_ds', default="results/rl_ds1.csv")
    parser.add_argument('--y_ds', default="results/labels.csv")
    parser.add_argument("--dataset_file", default="results/ds.csv")
    parser.add_argument("--n_inter", default=5, type=int) # we set it to 4 when --unaug_dataset = False
    parser.add_argument("--n_samples_per_inter", default=27321, type=int) # we set it to 43313 when --unaug_dataset = False

    parser.add_argument("--model", choices=['RL-GAN', 'AE+DQN'])
    parser.add_argument("--save_state_dict", default="results/ae1.pth")

    return parser.parse_args(args)

# discrete features
discrete = {
            "state": 5,
            # "service": 13,
            "ct_state_ttl": 6,
            # "dttl": 9,
            # "sttl": 13,
            "trans_depth": 11
}
# binary features
binary = ["proto", "is_ftp_login"]
# the rest are continuous features
discrete_and_binary = set(discrete.keys()).union(set(binary))
continuous = [feature for feature in main_u.X_train.columns if feature not in discrete_and_binary]

# When running the decoder
def types_append(decoder, discrete_out, continuous_out, binary_out, discrete_samples, continuous_samples, binary_samples):
    for feature in decoder.discrete_features:
        discrete_samples[feature].append(torch.argmax(torch.round(discrete_out[feature]), dim=-1))

    for feature in decoder.continuous_features:
        continuous_samples[feature].append(continuous_out[feature])

    for feature in decoder.binary_features:
        binary_samples[feature].append(torch.argmax(torch.round(binary_out[feature]), dim=-1))
    return discrete_samples, continuous_samples, binary_samples

def type_concat(decoder, discrete_samples, continuous_samples, binary_samples):
    for feature in decoder.discrete_features:
        discrete_samples[feature] = torch.cat(discrete_samples[feature], dim=0)

    for feature in decoder.continuous_features:
        continuous_samples[feature] = torch.cat(continuous_samples[feature], dim=0)

    for feature in decoder.binary_features:
        binary_samples[feature] = torch.cat(binary_samples[feature], dim=0)

    return discrete_samples, continuous_samples, binary_samples

def all_samples(discrete_samples, continuous_samples, binary_samples):
    discrete_tensors = list(discrete_samples.values())
    continuous_tensors = list(continuous_samples.values())
    binary_tensors = list(binary_samples.values())

    all_tensors = discrete_tensors + continuous_tensors + binary_tensors
    all_tensors = [t.unsqueeze(-1) if t.dim() == 1 else t for t in all_tensors]
    combined = torch.cat(all_tensors, dim=1)
    return combined

# Dataframe to Pytorch dataset
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample, label

# Train eval test split of pytorch dataset
def dataset_function(dataset, batch_size_t, batch_size_o, train=True):
    total_size = len(dataset)
    test_size = total_size // 5
    val_size = total_size // 10
    train_size = total_size - (test_size + val_size)
    train_subset = Subset(dataset, range(train_size))
    val_subset = Subset(dataset, range(train_size, train_size + val_size))
    test_subset = Subset(dataset, range(train_size + val_size, total_size))
    if train:
        train_loader = DataLoader(train_subset, batch_size=batch_size_t, shuffle=False)
        val_loader = DataLoader(val_subset, batch_size=batch_size_o, shuffle=False)
        return train_loader, val_loader

    else:
        test_loader = DataLoader(test_subset, batch_size=batch_size_o, shuffle=False)

        return test_loader

# inverse scaling (in benchmark classification)
def inverse_sc_cont(X, synth):
    synth_inv = synth * (X.max() - X.min()) + X.min()
    return pd.DataFrame(synth_inv, columns=X.columns, index=synth.index)

# choose unaugmented or augmented dataset
def dataset(original=False, train=True, confidence=True):
    if original:
        if train:
            dataset = CustomDataset(main_u.X_train_sc.to_numpy(), main_u.y_train.to_numpy())
        else:
            dataset = CustomDataset(main_u.X_test_sc.to_numpy(), main_u.y_test.to_numpy())
    else:
        args = parse_args(sys.argv[1:])
        df_org = pd.concat([main_u.X_sc, main_u.y], axis=1)
        X_rl = pd.DataFrame(pd.read_csv(f"{args.X_ds}"))
        X_rl = X_rl.apply(lambda col: col.str.strip("[]").astype(float) if col.dtype == "object" else col)
        y_rl = pd.DataFrame(pd.read_csv(f"{args.y_ds}"))
        df_rl = pd.concat([X_rl, y_rl], axis=1)
        df_rl = df_rl[df_rl["attack_cat"] != 2]
        if confidence:
            df_rl = df_rl[df_rl["confidence"] > 0.89]
            df_rl = df_rl.drop("confidence", axis=1)
        df = pd.concat([df_org, df_rl], axis=0)
        X = df.drop(["attack_cat"], axis=1)
        y = df["attack_cat"]

        X_train, X_test, y_train, y_test = main_u.vertical_split(X, y)
        if train:
            dataset = CustomDataset(X_train.to_numpy(), labels=y_train.to_numpy())
        else:
            dataset = CustomDataset(X_test.to_numpy(), labels=y_test.to_numpy())
    return dataset

# DRL iterator
class RL_dataloader:
    def __init__(self, dataloader):
        self.loader = dataloader
        self.loader_iter = iter(self.loader)

    def __len__(self):
        return len(self.loader)

    def next_data(self):
        try:
            data, label = next(self.loader_iter)

        except:
            self.loader_iter = iter(self.loader)
            data, label = next(self.loader_iter)

        return data, label

