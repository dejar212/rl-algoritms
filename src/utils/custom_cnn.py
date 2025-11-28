import gymnasium as gym
import torch as th
import torch.nn as nn
import numpy as np
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
    """
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

class CustomCombinedExtractor(BaseFeaturesExtractor):
    """
    Combined extractor for Dict observation spaces.
    Supports 'local', 'global', 'target', and legacy keys.
    """
    def __init__(self, observation_space: gym.spaces.Dict, cnn_output_dim: int = 256):
        super().__init__(observation_space, features_dim=1)
        
        extractors = {}
        total_concat_size = 0
        
        for key, subspace in observation_space.spaces.items():
            if key == "local":
                extractors[key] = CustomCNN(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
            elif key == "global":
                extractors[key] = CustomCNN(subspace, features_dim=64)
                total_concat_size += 64
            elif key == "target":
                # Vector (2,)
                extractors[key] = nn.Flatten()
                total_concat_size += subspace.shape[0]
            elif key == "image":
                extractors[key] = CustomCNN(subspace, features_dim=cnn_output_dim)
                total_concat_size += cnn_output_dim
            elif key == "vector":
                extractors[key] = nn.Flatten()
                total_concat_size += subspace.shape[0]
            else:
                extractors[key] = nn.Flatten()
                total_concat_size += np.prod(subspace.shape)
                
        self.extractors = nn.ModuleDict(extractors)
        self._features_dim = total_concat_size
        
    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        for key, extractor in self.extractors.items():
            encoded_tensor_list.append(extractor(observations[key]))
        return th.cat(encoded_tensor_list, dim=1)
