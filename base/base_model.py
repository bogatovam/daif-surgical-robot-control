import copy

import gym
import torch.nn as nn
import numpy as np
from abc import abstractmethod
import torch
from stable_baselines3.common.preprocessing import maybe_transpose, is_image_space
from stable_baselines3.common.utils import get_device, obs_as_tensor, is_vectorized_observation
from typing import Union
import warnings


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    def __init__(self):
        super().__init__()

    def set_training_mode(self, mode: bool) -> None:
        self.train(mode)

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def _get_constructor_parameters(self):
        return dict()

    def save(self, path: str) -> None:
        """
        Save model to a given location.

        :param path:
        """
        torch.save({"state_dict": self.state_dict(), "data": self._get_constructor_parameters()}, path)

    @classmethod
    def load(cls, path: str, device: Union[torch.device, str] = "auto") -> "BaseModel":
        """
        Load model from path.

        :param path:
        :param device: Device on which the policy should be loaded.
        :return:
        """
        device = get_device(device)
        saved_variables = torch.load(path, map_location=device)

        # Create policy object
        model = cls(**saved_variables["data"])  # pytype: disable=not-instantiable
        # Load weights
        model.load_state_dict(saved_variables["state_dict"])
        model.to(device)
        return model

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)
