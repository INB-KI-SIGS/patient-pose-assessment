import collections
from abc import ABC
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
import yaloader
from mllooper import State
from mllooper.data import DatasetState
from mllooper.metrics import ScalarMetric, ScalarMetricConfig
from mllooper.models import ModelState
from scipy.stats import spearmanr


class CorrectBy(ScalarMetric):
    def __init__(self, max_distance: float = 0.5, name: Optional[str] = None, **kwargs):
        name = f"{name}-{max_distance}"
        super().__init__(name=name, **kwargs)
        self.max_distance = max_distance

    def calculate_metric(self, state: State) -> torch.Tensor:
        dataset_state: DatasetState = state.dataset_state
        model_state: ModelState = state.model_state

        if "target" not in dataset_state.data:
            raise ValueError(
                f"{self.name} requires a tensor with the targets to be in "
                f"state.dataset_state.data['target']"
            )
        absolute_error = (
            model_state.output.squeeze() - dataset_state.data["target"].squeeze()
        ).abs()
        mean_correct = (absolute_error <= self.max_distance).float().mean()
        return mean_correct

    @torch.no_grad()
    def is_better(self, x, y) -> bool:
        return x.mean() > y.mean()


@yaloader.loads(CorrectBy)
class CorrectByConfig(ScalarMetricConfig):
    name: str = "CorrectBy"
    max_distance: float = 0.5


class CorrectCutAt(ScalarMetric):
    def __init__(self, cut_at: float = 2.0, name: Optional[str] = None, **kwargs):
        name = f"{name}-{cut_at}"
        super().__init__(name=name, **kwargs)
        self.cut_at = cut_at

    def calculate_metric(self, state: State) -> torch.Tensor:
        dataset_state: DatasetState = state.dataset_state
        model_state: ModelState = state.model_state

        if "target" not in dataset_state.data:
            raise ValueError(
                f"{self.name} requires a tensor with the targets to be in "
                f"state.dataset_state.data['target']"
            )
        label = dataset_state.data["target"].squeeze() > self.cut_at
        prediction = model_state.output.squeeze() > self.cut_at

        mean_correct = (label == prediction).float().mean()
        return mean_correct

    @torch.no_grad()
    def is_better(self, x, y) -> bool:
        return x.mean() > y.mean()


@yaloader.loads(CorrectCutAt)
class CorrectCutAtConfig(ScalarMetricConfig):
    name: str = "CorrectCutAt"
    cut_at: float = 2.5


class Correlation(ScalarMetric, ABC):
    def __init__(self, name="Correlation", **kwargs):
        super().__init__(name=name, **kwargs)

        self.predictions_per_dataset = defaultdict(lambda: collections.deque(maxlen=500))
        self.targets_per_dataset = defaultdict(lambda: collections.deque(maxlen=500))

    def calculate_metric(self, state: State) -> torch.Tensor:
        dataset_state: DatasetState = state.dataset_state
        model_state: ModelState = state.model_state

        if "target" not in dataset_state.data:
            raise ValueError(
                f"{self.name} requires a tensor with the targets to be in "
                f"state.dataset_state.data['target']"
            )

        targets = dataset_state.data["target"]
        predictions = model_state.output

        self.predictions_per_dataset[dataset_state.name].extend(map(float, predictions.cpu()))
        self.targets_per_dataset[dataset_state.name].extend(map(float, targets.cpu()))

        correlation = self.calculate_correlation(
            np.array(list(self.predictions_per_dataset[dataset_state.name])),
            np.array(list(self.targets_per_dataset[dataset_state.name])),
        )
        return torch.Tensor([correlation])

    def calculate_correlation(self, predictions, targets):
        raise NotImplementedError

    def is_better(self, x, y) -> bool:
        if y is None:
            return True
        return x.mean() > y.mean()


class SpearmanCorrelation(Correlation):
    def calculate_correlation(self, predictions, targets):
        return spearmanr(a=predictions, b=targets).correlation


@yaloader.loads(SpearmanCorrelation)
class SpearmanCorrelationConfig(ScalarMetricConfig):
    name: str = "SpearmanCorrelation"
