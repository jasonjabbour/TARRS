# Example Custom Policy Network Implemenation
# Tested on the Atari Breakout Environment

import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Bernoulli
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.utils import add_self_loops
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

import gymnasium as gym
from stable_baselines3 import PPO
import numpy as np
from gymnasium.envs.box2d import CarRacing
from stable_baselines3.common.env_util import make_vec_env


class CNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(CNNFeatureExtractor, self).__init__(observation_space, features_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten()
        ).to(self.device)

    def forward(self, observations):
        observations = observations.float() / 255.0  # normalize pixel values
        observations = observations.to(self.device)
        features = self.cnn(observations)
        return features


class CustomCNNActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, features_dim=256, **kwargs):
        # Call the super constructor first
        super(CustomCNNActorCriticPolicy, self).__init__(
            observation_space, action_space, lr_schedule,
            features_extractor_class=CNNFeatureExtractor,
            features_extractor_kwargs={'features_dim': features_dim},
            **kwargs
        )

        # Get the device dynamically based on CUDA availability
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Extract the size of the MultiDiscrete space, which equals the max number of nodes
        self.num_nodes = 4

        # Actor outputs logits for each node pair
        self.actor = nn.Sequential(
            nn.Linear(22528, features_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(features_dim * 2, self.num_nodes),
            nn.ReLU()
        ).to(self._device)

        self.critic = nn.Sequential(
            nn.Linear(22528, features_dim * 2),
            nn.ReLU(),
            nn.Linear(features_dim * 2, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, 1)
        ).to(self._device)

        # Assigning the actor network list to action_net, used by the base class.
        self.action_net = self.actor
        # Assigning the critic network to value_net, used by the base class.
        self.value_net = self.critic

        # Setup the optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr_schedule(1))

    def _build(self, lr_schedule):
        # This method intentionally does nothing to prevent the base class from building its own networks
        pass

    def predict_values(self, obs):
        """
        Get the estimated values according to the current policy given the observations.
        :param obs: Observation
        :return: the estimated values.
        """
        obs_tensor = obs.to(self._device) 
        # Ensure input tensors are on the right device
        features = self.features_extractor(obs_tensor)  
        # Execute model prediction
        return self.value_net(features) 
    
    def forward(self, obs, deterministic=False):

        # Check and convert inputs
        if isinstance(obs, tuple):
            # If the observation is a tuple, unpack it as needed
            obs = obs[0]

        # Ensure that each item in the observation dictionary is converted to a tensor and moved to the correct device
        obs_tensor = obs.to(self._device) 

        # Extract features using the custom feature extractor
        features = self.features_extractor(obs_tensor)  # Directly use the feature extractor here

        # Compute logits using the actor network and reshape them to match the action mask dimensions
        logits = self.actor(features)

        # Apply softmax to convert masked logits into probabilities and view as a flat vector instead of a matrix
        probabilities = F.softmax(logits, dim=-1) 

        # Create a categorical distribution from the probabilities to sample actions
        dist = torch.distributions.Categorical(probabilities)

        # Sample or select the single maximum probability action based on whether deterministic mode is on
        actions = dist.sample() if not deterministic else torch.argmax(probabilities, dim=1)

        # Fetch log probability handling any batch dimensions
        log_prob = dist.log_prob(actions).squeeze()

        # Evaluate the state value using the critic network
        values = self.critic(features).squeeze()

        # Return the actions, their log probabilities, and the state values
        return actions, values, log_prob
   
    def evaluate_actions(self, obs, actions):

        # Check and convert inputs
        if isinstance(obs, tuple):
            # If the observation is a tuple, unpack it as needed
            obs = obs[0]

        # Ensure that each item in the observation dictionary is converted to a tensor and moved to the correct device
        obs_tensor = obs.to(self._device) 

        # Extract features using the custom feature extractor
        features = self.features_extractor(obs_tensor)  # Directly use the feature extractor here

        # Compute logits using the actor network and reshape them to match the action mask dimensions
        logits = self.actor(features)

        # Apply softmax to convert masked logits into probabilities and view as a flat vector instead of a matrix
        probabilities = F.softmax(logits, dim=-1) 

        # Create a categorical distribution from the probabilities to sample actions
        dist = torch.distributions.Categorical(probabilities)

        # Calculate the log probability for the given actions
        log_probs = dist.log_prob(actions).squeeze()

        # Calculate the entropy of the distribution, a measure of randomness
        entropy = dist.entropy().squeeze()

        # Evaluate the state value using the critic network
        values = self.critic(features).squeeze()

        # Return the state values, log probabilities of the actions, and entropy
        return values, log_probs, entropy


    def get_distribution(self, obs):
        """Get the action distribution based on the policy network's output."""

        # Ensure that each item in the observation dictionary is converted to a tensor and moved to the correct device
        obs_tensor = obs.to(self._device) 

        # Extract features using the custom feature extractor
        features = self.features_extractor(obs_tensor)  # Directly use the feature extractor here

        # Compute logits using the actor network and reshape them to match the action mask dimensions
        logits = self.actor(features)

        # Apply softmax to convert masked logits into probabilities and view as a flat vector instead of a matrix
        probabilities = F.softmax(logits, dim=-1) 

        # Create a categorical distribution from the probabilities to sample actions
        distribution = torch.distributions.Categorical(probabilities)

        return distribution


    def _predict(self, obs, deterministic=False):
        """Predict actions based on the policy distribution and whether to use deterministic actions."""

        # Compute the distribution using the prepared observations
        distribution = self.get_distribution(obs)

        if deterministic:
            # If deterministic, choose the action with the highest probability
            action_indices = torch.argmax(distribution.probs, dim=1)
        else:
            # Otherwise, sample from the distribution
            action_indices = distribution.sample()

        return action_indices
    

if __name__ == '__main__':

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    vec_env = make_vec_env("Breakout-v0", n_envs=8)

    model = PPO(CustomCNNActorCriticPolicy, vec_env, verbose=1)
    model.learn(total_timesteps=500000)

    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        vec_env.render("human")
