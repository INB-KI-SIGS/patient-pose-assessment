from typing import Any, Dict, Optional, List

import torch
import yaloader
from mllooper import State
from mllooper.data import DatasetState
from mllooper.models import ModelConfig, Model
from mllooper.models.efficientnet.model import b0
from torch import nn


class BranchDropout(nn.Module):
    # see https://stackoverflow.com/questions/54109617/implementing-dropout-from-scratch

    def __init__(self, p: float = 0.5):
        super(BranchDropout, self).__init__()
        if p <= 0 or p >= 1:
            raise ValueError(f"dropout probability has to be between 0 and 1, but got {p}")
        self.p = p

    def forward(self, x: List[torch.Tensor]):
        if self.training and self.p != 0:
            binomial = torch.distributions.binomial.Binomial(probs=1 - self.p)
            keeps = binomial.sample((len(x),))
            if torch.count_nonzero(keeps) == 0:
                keeps[torch.randint(len(x), (1,))] = 1

            # Use current mean for scaling instead of q
            # because current mean and q are always different due tue small sample size
            return [model_output * keep * (1.0 / keeps.mean()) for model_output, keep in zip(x, keeps)]
        return x


class FusionBaseModule(nn.Module):
    def __init__(
        self,
        shared_weights: bool,
        num_classes: int,
        in_channels: int = 1,
        dropout: float = 0.2,
        stochastic_depth_prob: float = 0.2,
        branch_dropout: Optional[float] = None,
        pretrained: bool = False,
    ):
        super().__init__()
        self.shared_weights = shared_weights

        if self.shared_weights:
            self.module = b0(
                pretrained=pretrained,
                in_channels=in_channels,
                num_classes=num_classes,
                dropout=dropout,
                stochastic_depth_prob=stochastic_depth_prob,
            )
        else:
            self.cam_left_module = b0(
                pretrained=pretrained,
                in_channels=in_channels,
                num_classes=num_classes,
                dropout=dropout,
                stochastic_depth_prob=stochastic_depth_prob,
            )
            self.cam_right_module = b0(
                pretrained=pretrained,
                in_channels=in_channels,
                num_classes=num_classes,
                dropout=dropout,
                stochastic_depth_prob=stochastic_depth_prob,
            )

        self.branch_dropout = None if branch_dropout is None else BranchDropout(p=branch_dropout)

    def get_cam_features(self, cam_left_image, cam_right_image):
        if self.shared_weights:
            images = torch.cat((cam_left_image, cam_right_image), dim=0)
            features = self.module(images)

            features_cam_left, features_cam_right = torch.split(
                features, [cam_left_image.shape[0], cam_right_image.shape[0]]
            )
        else:
            features_cam_left = self.cam_left_module(cam_left_image)
            features_cam_right = self.cam_right_module(cam_right_image)

        return features_cam_left, features_cam_right

    def forward(self, cam_left_image, cam_right_image):
        raise NotImplementedError


class LateFusionModule(FusionBaseModule):
    def __init__(
        self,
        num_classes: int,
        outer_dropout: Optional[float] = None,
        extra_fusion_features: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(num_classes=num_classes, **kwargs)
        if extra_fusion_features is not None:
            self.fc = nn.Sequential(
                nn.Linear(in_features=num_classes * 2, out_features=extra_fusion_features),
                nn.SiLU(),
                nn.Identity() if outer_dropout is None else nn.Dropout(p=outer_dropout),
                nn.Linear(in_features=extra_fusion_features, out_features=1),
            )
        else:
            self.fc = nn.Sequential(
                nn.Identity() if outer_dropout is None else nn.Dropout(p=outer_dropout),
                nn.Linear(in_features=num_classes * 2, out_features=1),
            )

    def forward(self, cam_left_image, cam_right_image):
        features_cam_left, features_cam_right = self.get_cam_features(cam_left_image, cam_right_image)

        if self.branch_dropout is not None:
            features_cam_left, features_cam_right = self.branch_dropout(
                [features_cam_left, features_cam_right]
            )

        fusion_features = torch.cat([features_cam_left, features_cam_right], dim=1)
        fusion_features = torch.functional.F.silu(fusion_features)

        quality_prediction = self.fc(fusion_features)
        return quality_prediction


class ScoreFusionModule(FusionBaseModule):
    def __init__(self, **kwargs):
        super().__init__(num_classes=1, **kwargs)

    def forward(self, cam_left_image, cam_right_image):
        quality_cam_left, quality_cam_right = self.get_cam_features(cam_left_image, cam_right_image)

        if self.branch_dropout is not None:
            quality_cam_left, quality_cam_right = self.branch_dropout([quality_cam_left, quality_cam_right])

        quality_prediction = (quality_cam_left + quality_cam_right) / 2
        return quality_prediction


class FusionNetworkBase(Model):
    def step(self, state: State) -> None:
        dataset_state: DatasetState = state.dataset_state
        self.state.output = None

        cam_left_image, cam_right_image = self.format_module_input(dataset_state.data)

        self.module.train() if dataset_state.train else self.module.eval()
        self._parallel_module.train() if dataset_state.train else self._parallel_module.eval()
        with torch.set_grad_enabled(dataset_state.train):
            module_output = self.module(cam_left_image, cam_right_image)

        self.state.output = self.format_module_output(module_output)
        state.model_state = self.state

    @staticmethod
    def format_module_input(data: Any) -> Any:
        if (
            isinstance(data, Dict)
            and "input_cam_left" in data.keys()
            and isinstance(data["input_cam_left"], torch.Tensor)
            and "input_cam_right" in data.keys()
            and isinstance(data["input_cam_right"], torch.Tensor)
        ):
            return data["input_cam_left"], data["input_cam_right"]
        else:
            raise NotImplementedError

    @staticmethod
    def format_module_output(output: Any) -> Any:
        if isinstance(output, torch.Tensor):
            return output
        else:
            raise NotImplementedError


class LateFusionNetwork(FusionNetworkBase):
    def __init__(
        self,
        shared_weights: bool,
        in_channels: int = 1,
        num_classes: int = 10,
        dropout: float = 0.01,
        outer_dropout: Optional[float] = None,
        stochastic_depth_prob: float = 0.5,
        branch_dropout: Optional[float] = None,
        extra_fusion_features: Optional[int] = None,
        pretrained: bool = False,
        **kwargs,
    ):
        torch_model = LateFusionModule(
            shared_weights=shared_weights,
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout,
            outer_dropout=outer_dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            branch_dropout=branch_dropout,
            extra_fusion_features=extra_fusion_features,
            pretrained=pretrained,
        )
        super().__init__(torch_model, **kwargs)


class ScoreFusionNetwork(FusionNetworkBase):
    def __init__(
        self,
        shared_weights: bool,
        in_channels: int = 1,
        dropout: float = 0.01,
        stochastic_depth_prob: float = 0.5,
        branch_dropout: Optional[float] = None,
        pretrained: bool = False,
        **kwargs,
    ):
        torch_model = ScoreFusionModule(
            shared_weights=shared_weights,
            in_channels=in_channels,
            dropout=dropout,
            stochastic_depth_prob=stochastic_depth_prob,
            branch_dropout=branch_dropout,
            pretrained=pretrained,
        )
        super().__init__(torch_model, **kwargs)


@yaloader.loads(None)
class FusionNetworkBaseConfig(ModelConfig):
    in_channels: int = 1
    dropout: float = 0.1
    stochastic_depth_prob: float = 0.6
    branch_dropout: Optional[float] = None
    pretrained: bool = False
    shared_weights: bool


@yaloader.loads(LateFusionNetwork)
class LateFusionNetworkConfig(FusionNetworkBaseConfig):
    name: str = "Late Fusion"
    num_classes: int = 3
    extra_fusion_features: Optional[int] = None
    outer_dropout: Optional[float] = None
    pretrained: bool = False

    shared_weights: bool = False


@yaloader.loads(LateFusionNetwork)
class LateFusionNetworkSharedWeightsConfig(LateFusionNetworkConfig):
    name: str = "Late Fusion Shared Weights"
    shared_weights: bool = True


@yaloader.loads(ScoreFusionNetwork)
class ScoreFusionNetworkConfig(FusionNetworkBaseConfig):
    name: str = "Score Fusion"
    shared_weights: bool = False


@yaloader.loads(ScoreFusionNetwork)
class ScoreFusionNetworkSharedWeightsConfig(ScoreFusionNetworkConfig):
    name: str = "Score Fusion Shared Weights"
    shared_weights: bool = True
