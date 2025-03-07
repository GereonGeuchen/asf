try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import pandas as pd
from sklearn.impute import SimpleImputer

from asf.predictors.abstract_predictor import AbstractPredictor
from asf.predictors.utils.datasets import RegressionDataset
from asf.predictors.utils.mlp import get_mlp


class RegressionMLP(AbstractPredictor):
    def __init__(
        self,
        model: torch.nn.Module | None = None,
        input_size: int | None = None,
        loss: torch.nn.modules.loss._Loss | None = torch.nn.MSELoss(),
        optimizer: torch.optim.Optimizer | None = torch.optim.Adam,
        batch_size: int = 128,
        epochs: int = 2000,
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

    def _get_dataloader(self, features: pd.DataFrame, performance: pd.DataFrame):
        dataset = RegressionDataset(features, performance)
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

        features = pd.DataFrame(
            SimpleImputer().fit_transform(features.values),
            index=features.index,
            columns=features.columns,
        )
        dataloader = self._get_dataloader(features, performance)

        optimizer = self.optimizer(self.model.parameters())
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for i, (X, y) in enumerate(dataloader):
                X, y = X.to(self.device), y.to(self.device)
                y = y.unsqueeze(-1)
                optimizer.zero_grad()
                y_pred = self.model(X)
                loss = self.loss(y_pred, y)
                total_loss += loss.item()
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
        self.model.eval()

        features = torch.from_numpy(features.values).to(self.device)
        predictions = self.model(features).detach().numpy()

        return predictions

    def save(self, file_path):
        torch.save(self.model, file_path)

    def load(self, file_path):
        torch.load(file_path)
