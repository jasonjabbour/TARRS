import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Bernoulli
import torch.nn.functional as F
import torch_geometric.nn as gnn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class GNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(GNNFeatureExtractor, self).__init__(observation_space, features_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the number of features each node has
        # initially just the attacked status
        node_feature_dim = 1
        self.gcn_full = gnn.GCNConv(node_feature_dim, features_dim // 2).to(self.device)
        self.gcn_mst = gnn.GCNConv(node_feature_dim, features_dim // 2).to(self.device)

    def forward(self, observations):

        # Extract relevant matrices and vectors
        full_network = observations['full_network'][0].to(self.device)
        mst_status = observations['mst'][0].to(self.device)
        attacked_status = observations['attacked'][0].to(self.device)
        weights = observations['weights'][0].to(self.device)

        # Creating edge indices from adjacency matrices
        edge_index_full = (full_network > 0).nonzero(as_tuple=False).t().contiguous()
        edge_weights_full = weights[full_network > 0].clone().detach().float()

        edge_index_mst = (mst_status > 0).nonzero(as_tuple=False).t().contiguous()
        edge_weights_mst = weights[mst_status > 0].clone().detach().float()

        # Prepare node features: here we use only the attacked status as node features for simplicity
        node_features = attacked_status.unsqueeze(-1).float()  # Assuming attacked_status is a [N,] vector

        # Passing node features, edge indices, and edge weights to GCN layers
        x_full = self.gcn_full(node_features, edge_index_full, edge_weights_full)
        x_mst = self.gcn_mst(node_features, edge_index_mst, edge_weights_mst)
  
        # Applying non-linearity (ReLU)
        x_full = F.relu(x_full)
        x_mst = F.relu(x_mst)

        # Pooling the features from both GCNs
        x_full_pooled = gnn.global_mean_pool(x_full, batch=torch.zeros(x_full.size(0), dtype=torch.long, device=self.device))
        x_mst_pooled = gnn.global_mean_pool(x_mst, batch=torch.zeros(x_mst.size(0), dtype=torch.long, device=self.device))

        # Concatenate features from both GCN outputs
        combined_features = torch.cat([x_full_pooled, x_mst_pooled], dim=-1)

        return combined_features

class CustomGNNActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, features_dim=256, **kwargs):
        # Call the super constructor first
        super(CustomGNNActorCriticPolicy, self).__init__(
            observation_space, action_space, lr_schedule,
            features_extractor_class=GNNFeatureExtractor,
            features_extractor_kwargs={'features_dim': features_dim},
            **kwargs
        )

        # Get the device dynamically based on CUDA availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize actor and critic networks
        self.actor = nn.Sequential(
            nn.Linear(features_dim, action_space.n),
            nn.Sigmoid()
        ).to(device)
        
        self.critic = nn.Sequential(
            nn.Linear(features_dim, 1)
        ).to(device)

        # Assign to action_net and value_net which are used by the base class
        self.action_net = self.actor
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
        features = self.features_extractor(obs)
        return self.value_net(features)
    
    def forward(self, obs, deterministic=False):
        assert all(key in obs for key in ['full_network', 'mst', 'attacked', 'weights']), "Missing keys in observations"

        # Ensuring all entries are tensors and are on the correct device
        device = next(self.features_extractor.parameters()).device  # Get the device from the parameters of the features extractor

        # Check and convert inputs
        if isinstance(obs, tuple):
            # If the observation is a tuple, unpack it as needed
            obs = obs[0]

        # Ensure that each item in the observation dictionary is converted to a tensor and moved to the correct device
        obs_tensors = {k: v.to(device) if torch.is_tensor(v) else torch.tensor(v, dtype=torch.float32).to(device) for k, v in obs.items()}

        # Extract features using the custom feature extractor
        features = self.features_extractor(obs_tensors)  # Directly use the feature extractor here

        # Generate actions from the neural networks
        action_probs = self.actor(features)

        # Bernoulli distribution for binary actions
        dist = Bernoulli(action_probs) 
        # Deterministic thresholding at 0.5 
        actions = dist.sample() if not deterministic else (action_probs > 0.5).float()  

        # Compute log probabilities of the selected actions
        log_probs = dist.log_prob(actions).sum(dim=1)  # Sum log probs for all actions if they are independent
        
        # Get the critic value
        values = self.critic(features)

        return actions, values, log_probs

    def evaluate_actions(self, obs, actions):
        # Extract features using the GNN features extractor
        features = self.features_extractor(obs)

        # Compute the policy's output (latent_pi) and the value function's output (latent_vf)
        latent_pi = self.actor(features)
        latent_vf = self.critic(features)

        # Create a distribution for calculating log probabilities and entropy
        dist = torch.distributions.Bernoulli(logits=latent_pi)

        # Calculate log probabilities of the taken actions and entropy
        log_prob = dist.log_prob(actions).sum(dim=1, keepdim=True)
        entropy = dist.entropy().sum(dim=1, keepdim=True)

        # The critic's output is the estimated value of the state
        values = latent_vf

        return values, log_prob, entropy

    def get_distribution(self, obs):
        """Get the action distribution based on the policy network's output."""
        features = self.extract_features(obs)  # Extract features from observations
        action_probs = self.actor(features)  # Get action probabilities
        return torch.distributions.Bernoulli(probs=action_probs)  # Return Bernoulli distribution based on the probabilities

    def _predict(self, obs, deterministic=False):
        """Predict actions based on the policy distribution and whether to use deterministic actions."""
        distribution = self.get_distribution(obs)
        if deterministic:
            # Deterministic actions: choose action based on probability threshold
            return (distribution.probs >= 0.5).float()
        else:
            # Stochastic actions: sample from the Bernoulli distribution
            return distribution.sample()


