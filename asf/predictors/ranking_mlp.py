from typing import Callable

import pandas as pd

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from asf.predictors.abstract_predictor import AbstractPredictor
from asf.predictors.utils.datasets import RankingDataset
from asf.predictors.utils.losses import bpr_loss
from asf.predictors.utils.mlp import get_mlp


class RankingMLP(AbstractPredictor):
    def __init__(
        self,
        model: torch.nn.Module | None = None,
        input_size: int | None = None,
        loss: Callable | None = bpr_loss,
        optimizer: torch.optim.Optimizer | None = torch.optim.Adam,
        batch_size: int = 128,
        epochs: int = 500,
        seed: int = 42,
        device: str = "cpu",
        compile=True,
        **kwargs,
    ):
        """
        Initializes the JointRanking with the given parameters.

        Args:
            model: The model to be used.
        """
        super().__init__(**kwargs)
        assert TORCH_AVAILABLE, "PyTorch is not available. Please install it."

        assert model is not None or input_size is not None, (
            "Either model or input_size must be provided."
        )

        torch.manual_seed(seed)

        if model is None:
            self.model = get_mlp(input_size=input_size, output_size=1)
        else:
            self.model = model

        self.model.to(device)
        self.device = device

        self.loss = loss
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.epochs = epochs

        if compile:
            self.model = torch.compile(self.model)

    def _get_dataloader(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        algorithm_features: pd.DataFrame,
    ):
        dataset = RankingDataset(features, performance, algorithm_features)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def fit(
        self,
        features: pd.DataFrame,
        performance: pd.DataFrame,
        algorithm_features: pd.DataFrame,
    ):
        """
        Fits the model to the given feature and performance data.

        Args:
            features: DataFrame containing the feature data.
            performance: DataFrame containing the performance data.
        """

        print(self.model)
        dataloader = self._get_dataloader(features, performance, algorithm_features)

        optimizer = self.optimizer(self.model.parameters())
        self.model.train()
        for epoch in range(1000):  # self.epochs):
            total_loss = 0
            for i, ((Xc, Xs, Xl), (yc, ys, yl)) in enumerate(dataloader):
                Xc, Xs, Xl = Xc.to(self.device), Xs.to(self.device), Xl.to(self.device)
                yc, ys, yl = yc.to(self.device), ys.to(self.device), yl.to(self.device)

                yc = yc.float().unsqueeze(1)
                ys = ys.float().unsqueeze(1)
                yl = yl.float().unsqueeze(1)

                # yc = torch.log10(yc)
                # ys = torch.log10(ys)
                # yl = torch.log10(yl)
                # print(Xc)
                # print(Xs)
                # print(Xl)

                # print(yc)
                # print(ys)
                # print(yl)
                optimizer.zero_grad()

                y_pred = self.model(Xc)
                y_pred_s = self.model(Xs)
                y_pred_l = self.model(Xl)

                loss = self.loss(y_pred, y_pred_s, y_pred_l, yc, ys, yl)
                # loss = torch.nn.functional.mse_loss(y_pred, yc)
                total_loss += loss.item()

                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch}, Loss: {total_loss / len(dataloader)}")

        return self

    def predict(self, features: pd.DataFrame):
        """
        Predicts the performance of algorithms for the given features.

        Args:
            features: DataFrame containing the feature data.

        Returns:
            DataFrame containing the predicted performance data.
        """
        self.model.eval()

        features = torch.from_numpy(features.values).to(self.device).float()
        predictions = self.model(features).detach().numpy()

        return predictions

    def save(self, file_path):
        torch.save(self.model, file_path)

    def load(self, file_path):
        torch.load(file_path)
