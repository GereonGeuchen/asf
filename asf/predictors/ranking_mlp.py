import torch
from asf.predictors.utils.datasets import RankingDataset, RegressionDataset
from asf.predictors.utils.mlp import get_mlp
from asf.predictors.utils.losses import bpr_loss
from asf.predictors.abstract_predictor import AbstractPredictor
import pandas as pd
from typing import Callable


class RankingMLP(AbstractPredictor):
    def __init__(
        self,
        model: torch.nn.Module | None = None,
        input_size: int | None = None,
        loss: Callable | None = bpr_loss,
        optimizer: torch.optim.Optimizer | None = torch.optim.Adam,
        batch_size: int = 128,
        epochs: int = 10,
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

    def _get_dataloader(self, features: pd.DataFrame, performance: pd.DataFrame):
        dataset = RankingDataset(features, performance)
        return torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

    def fit(self, features: pd.DataFrame, performance: pd.DataFrame):
        """
        Fits the model to the given feature and performance data.

        Args:
            features: DataFrame containing the feature data.
            performance: DataFrame containing the performance data.
        """

        dataloader = self._get_dataloader(features, performance)

        optimizer = self.optimizer(self.model.parameters())

        for epoch in range(self.epochs):
            for i, ((Xc, Xs, Xl), (yc, ys, yl)) in enumerate(dataloader):
                Xc, Xs, Xl = Xc.to(self.device), Xs.to(self.device), Xl.to(self.device)
                yc, ys, yl = yc.to(self.device), ys.to(self.device), yl.to(self.device)

                optimizer.zero_grad()

                y_pred = self.model(Xc)
                y_pred_s = self.model(Xs)
                y_pred_l = self.model(Xl)

                loss = self.loss(y_pred, y_pred_s, y_pred_l, yc, ys, yl)

                loss.backward()
                optimizer.step()

        return self

    def predict(self, features: pd.DataFrame):
        """
        Predicts the performance of algorithms for the given features.

        Args:
            features: DataFrame containing the feature data.

        Returns:
            DataFrame containing the predicted performance data.
        """
        dataset = RegressionDataset(features)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False
        )

        predictions = []
        for i, X in enumerate(dataloader):
            X = X.to(self.device)
            y_pred = self.model(X)
            predictions.append(y_pred)

        return pd.concat(predictions)

    def save(self, file_path):
        torch.save(self.model, file_path)

    def load(self, file_path):
        torch.load(file_path)
