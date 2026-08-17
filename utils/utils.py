from scipy import stats
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch

cuda = True if torch.cuda.is_available() else False


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


# Dataframe to Pytorch src
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

# Train eval test split of pytorch src
def dataset_function(ID_dataset, OOD, batch_size, batch_size_o, train=True):
    # total_size = len(dataset)
    # test_size = total_size // 5
    # train_size = total_size - test_size
    # train_subset = Subset(dataset, range(train_size))
    # test_subset = Subset(dataset, range(train_size, total_size))
    # ood_dataset = SyntheticOODDataset(X, regenerate_fn=make_ood_batch)
    # ood_test_dataset = SyntheticOODDataset(X, regenerate_fn=make_ood_batch)

    if train:
        train_loader = DataLoader(ID_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
        train_loader_ood = DataLoader(OOD, batch_size=batch_size_o, shuffle=True, num_workers=4, pin_memory=False)
        return train_loader, train_loader_ood

    else:
        test_loader = DataLoader(ID_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
        ood_test_loader = DataLoader(OOD, batch_size=batch_size_o, shuffle=False)
        return test_loader, ood_test_loader


def get_worst_attacks(out_score, fn_indices, ood_dataset, n=10):
    fn_scores = out_score[fn_indices]
    sorted_order = np.argsort(fn_scores)

    worst_indices = fn_indices[sorted_order][:n]
    worst_scores = fn_scores[sorted_order][:n]

    worst_labels = np.array([
        ood_dataset[i][-1] for i in worst_indices
    ])

    return worst_indices, worst_scores, worst_labels