import torch
import pandas as pd


class RegressionDataset(torch.utils.data.Dataset):
    def __init__(self, features, performance, dtype=torch.float32):
        self.features = torch.from_numpy(features.sort_index().to_numpy()).to(dtype)
        self.performance = torch.from_numpy(performance.sort_index().to_numpy()).to(
            dtype
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.performance[idx]


class RankingDataset(torch.utils.data.Dataset):
    def __init__(
        self, features: pd.DataFrame, performance: pd.DataFrame, dtype=torch.float32
    ):
        self.features = torch.from_numpy(features.sort_index().to_numpy()).to(dtype)
        self.performance = torch.from_numpy(performance.sort_index().to_numpy()).to(
            dtype
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        idx_perf = self.performance[idx]

        smaller = torch.argwhere(idx_perf > self.performance)
        smaller = smaller[torch.randint(0, len(smaller), 1)]
        larger = torch.argwhere(idx_perf < self.performance)
        larger = larger[torch.randint(0, len(larger), 1)]

        return (self.features[idx], self.features[smaller], self.features[larger]), (
            self.performance[idx],
            self.performance[smaller],
            self.performance[larger],
        )
