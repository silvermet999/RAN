from scipy import stats
import pandas as pd
from torch.utils.data import Dataset, DataLoader

import numpy as np
import torch

cuda = True if torch.cuda.is_available() else False


# Dataframe to Pytorch src
class CustomDataset(Dataset):
    def __init__(self, data, labels, slice_len=32):
        self.data = data
        self.labels = labels
        self.slice_len = slice_len

    def __len__(self):
        return len(self.data) - self.slice_len + 1

    def __getitem__(self, idx):
        sample = self.data[idx: idx + self.slice_len]
        label = torch.tensor(self.labels[idx + self.slice_len - 1], dtype=torch.long)
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
        train_loader = DataLoader(ID_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
        train_loader_ood = DataLoader(OOD, batch_size=batch_size_o, shuffle=False, num_workers=4, pin_memory=False)
        return train_loader, train_loader_ood

    else:
        test_loader = DataLoader(ID_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
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