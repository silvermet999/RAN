import argparse
import sys
from scipy import stats
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset

import numpy as np
import torch
import prep

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


# OOD
def shuffle_marginals(df, random_state=None):
    rng = np.random.default_rng(random_state)
    df_shuffled = df.copy()
    for col in df.columns:
        df_shuffled[col] = rng.permutation(df[col].values)
    return df_shuffled


def sample_from_marginals(df, n_samples, random_state=None):
    rng = np.random.default_rng(random_state)
    synthetic = {}
    for col in df.columns:
        values = df[col].values
        if pd.api.types.is_numeric_dtype(values):
            kde = stats.gaussian_kde(values)
            synthetic[col] = kde.resample(n_samples, seed=random_state).flatten()
        else:
            probs = df[col].value_counts(normalize=True)
            synthetic[col] = rng.choice(probs.index, size=n_samples, p=probs.values)
    return pd.DataFrame(synthetic)


def add_gaussian_noise(df, noise_scale=1.0, random_state=None):
    rng = np.random.default_rng(random_state)
    df_noisy = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    for col in numeric_cols:
        std = df[col].std()
        noise = rng.normal(0, noise_scale * std, size=len(df))
        df_noisy[col] = df[col] + noise
    return df_noisy

def make_ood_batch(df, random_state=None):
    rng = np.random.default_rng(random_state)
    n = len(df)
    third = n // 3

    shuffled = shuffle_marginals(df.iloc[:third], random_state=random_state)
    noisy = add_gaussian_noise(df.iloc[third:2*third], noise_scale=0.5, random_state=random_state)
    mixed = shuffle_marginals(df.iloc[2*third:], random_state=random_state)
    mixed = add_gaussian_noise(mixed, noise_scale=0.3, random_state=random_state)

    return pd.concat([shuffled, noisy, mixed], ignore_index=True)

class SyntheticOODDataset(Dataset):
    def __init__(self, id_df, regenerate_fn):
        self.id_df = id_df
        self.regenerate_fn = regenerate_fn
        self.offset = 0
        self._regenerate()

    def _regenerate(self):
        synthetic_df = self.regenerate_fn(self.id_df)
        values = synthetic_df.values.astype(np.float32)
        self.data = torch.tensor(values, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], -1


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
def dataset_function(dataset, X, batch_size_t, batch_size_o, train=True):
    total_size = len(dataset)
    test_size = total_size // 5
    train_size = total_size - test_size
    train_subset = Subset(dataset, range(train_size))
    test_subset = Subset(dataset, range(train_size, total_size))
    ood_dataset = SyntheticOODDataset(X, regenerate_fn=make_ood_batch)
    ood_test_dataset = SyntheticOODDataset(X, regenerate_fn=make_ood_batch)

    if train:
        train_loader = DataLoader(train_subset, batch_size=batch_size_t, shuffle=True, num_workers=4, pin_memory=False)
        train_loader_ood = DataLoader(ood_dataset, batch_size=batch_size_o, shuffle=True, num_workers=4, pin_memory=False)
        return train_loader, train_loader_ood

    else:
        test_loader = DataLoader(test_subset, batch_size=batch_size_t, shuffle=True, num_workers=4, pin_memory=False)
        ood_test_loader = DataLoader(ood_test_dataset, batch_size=batch_size_o, shuffle=False)
        return test_loader, ood_test_loader

# inverse scaling (in benchmark classification)
def inverse_sc_cont(X, synth):
    synth_inv = synth * (X.max() - X.min()) + X.min()
    return pd.DataFrame(synth_inv, columns=X.columns, index=synth.index)

# choose unaugmented or augmented dataset
# def dataset(original=False, train=True, confidence=True):
#     if original:
#         if train:
#             dataset = CustomDataset(prep.X_train_sc.to_numpy(), prep.y_train.to_numpy())
#         else:
#             dataset = CustomDataset(prep.X_test_sc.to_numpy(), prep.y_test.to_numpy())
#     else:
#         args = parse_args(sys.argv[1:])
#         df_org = pd.concat([prep.X_sc, prep.y], axis=1)
#         X_rl = pd.DataFrame(pd.read_csv(f"{args.X_ds}"))
#         X_rl = X_rl.apply(lambda col: col.str.strip("[]").astype(float) if col.dtype == "object" else col)
#         y_rl = pd.DataFrame(pd.read_csv(f"{args.y_ds}"))
#         df_rl = pd.concat([X_rl, y_rl], axis=1)
#         df_rl = df_rl[df_rl["attack_cat"] != 2]
#         if confidence:
#             df_rl = df_rl[df_rl["confidence"] > 0.89]
#             df_rl = df_rl.drop("confidence", axis=1)
#         df = pd.concat([df_org, df_rl], axis=0)
#         X = df.drop(["attack_cat"], axis=1)
#         y = df["attack_cat"]
#
#         X_train, X_test, y_train, y_test = prep.vertical_split(X, y)
#         if train:
#             dataset = CustomDataset(X_train.to_numpy(), labels=y_train.to_numpy())
#         else:
#             dataset = CustomDataset(X_test.to_numpy(), labels=y_test.to_numpy())
#     return dataset
