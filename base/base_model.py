import torch.nn as nn
import numpy as np
from abc import abstractmethod
import torch
from stable_baselines3.common.utils import get_device, is_vectorized_observation, obs_as_tensor
from typing import Any, Dict, List, Optional, Tuple, Type, Union
import warnings


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

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

        # Allow to load policy saved with older version of SB3
        if "sde_net_arch" in saved_variables["data"]:
            warnings.warn(
                "sde_net_arch is deprecated, please downgrade to SB3 v1.2.0 if you need such parameter.",
                DeprecationWarning,
            )
            del saved_variables["data"]["sde_net_arch"]

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
