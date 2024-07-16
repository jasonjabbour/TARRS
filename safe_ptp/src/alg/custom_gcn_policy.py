import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Bernoulli
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.utils import add_self_loops
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class GNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(GNNFeatureExtractor, self).__init__(observation_space, features_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the number of features each node has
        node_feature_dim = 2  # Includes attacked status and node index
        self.gcn_full = gnn.GCNConv(node_feature_dim, features_dim // 2).to(self.device)
        self.gcn_mst = gnn.GCNConv(node_feature_dim, features_dim // 2).to(self.device)

    # def forward(self, observations):
    #     # Removing the batch dimension by squeezing if it is of size 1
    #     node_features = observations['node_features'].squeeze(0).to(self.device).float()
    #     physical_edge_indices = observations['physical_edge_indices'].squeeze(0).to(self.device).long()
    #     physical_edge_weights = observations['physical_edge_weights'].squeeze(0).to(self.device).float()
    #     physical_edge_mask = observations['physical_edge_mask'].squeeze(0).to(self.device).bool()

    #     spanning_edge_indices = observations['spanning_tree_edge_indices'].squeeze(0).to(self.device).long()
    #     spanning_edge_weights = observations['spanning_tree_edge_weights'].squeeze(0).to(self.device).float()
    #     spanning_edge_mask = observations['spanning_tree_edge_mask'].squeeze(0).to(self.device).bool()

    #     # Apply Mask
    #     valid_physical_mask = physical_edge_mask.squeeze()
    #     valid_spanning_mask = spanning_edge_mask.squeeze()

    #     # Filter indices and weights using the boolean masks
    #     valid_physical_indices = physical_edge_indices[valid_physical_mask]
    #     valid_physical_weights = physical_edge_weights[valid_physical_mask]

    #     valid_spanning_indices = spanning_edge_indices[valid_spanning_mask]
    #     valid_spanning_weights = spanning_edge_weights[valid_spanning_mask]

    #     # Add self-loops
    #     num_nodes = node_features.size(0)

    #     # Transpose indices to shape [2, E]
    #     valid_physical_indices = valid_physical_indices.t()
    #     valid_spanning_indices = valid_spanning_indices.t()

    #     valid_physical_indices, valid_physical_weights = add_self_loops(
    #         valid_physical_indices, valid_physical_weights, fill_value=1.0, num_nodes=num_nodes)
    #     valid_spanning_indices, valid_spanning_weights = add_self_loops(
    #         valid_spanning_indices, valid_spanning_weights, fill_value=1.0, num_nodes=num_nodes)

    #     # Forward pass through GCNs for both the physical network and the spanning tree
    #     x_full = self.gcn_full(node_features, valid_physical_indices, valid_physical_weights)
    #     x_mst = self.gcn_mst(node_features, valid_spanning_indices, valid_spanning_weights)

    #     # Apply ReLU activation function
    #     x_full = F.relu(x_full)
    #     x_mst = F.relu(x_mst)

    #     # Global mean pooling to aggregate features
    #     x_full_pooled = gnn.global_mean_pool(x_full, batch=torch.zeros(x_full.size(0), dtype=torch.long, device=self.device))
    #     x_mst_pooled = gnn.global_mean_pool(x_mst, batch=torch.zeros(x_mst.size(0), dtype=torch.long, device=self.device))

    #     # Concatenate pooled features from both networks
    #     combined_features = torch.cat([x_full_pooled, x_mst_pooled], dim=-1)

    #     return combined_features

    def forward(self, observations):
        # Keep batch dimensions intact, move to device and cast types
        node_features = observations['node_features'].to(self.device).float()
        physical_edge_indices = observations['physical_edge_indices'].to(self.device).long()
        physical_edge_weights = observations['physical_edge_weights'].to(self.device).float()
        physical_edge_mask = observations['physical_edge_mask'].to(self.device).bool()

        spanning_edge_indices = observations['spanning_tree_edge_indices'].to(self.device).long()
        spanning_edge_weights = observations['spanning_tree_edge_weights'].to(self.device).float()
        spanning_edge_mask = observations['spanning_tree_edge_mask'].to(self.device).bool()

        combined_features_list = []

        for idx in range(node_features.size(0)):
            nf = node_features[idx]  # Node features for one graph instance
            num_nodes = nf.size(0)

            # Physical Network Processing
            pei = physical_edge_indices[idx]
            pew = physical_edge_weights[idx]
            pem = physical_edge_mask[idx].squeeze(-1)

            # Spanning Tree Network Processing
            sei = spanning_edge_indices[idx]
            sew = spanning_edge_weights[idx]
            sem = spanning_edge_mask[idx].squeeze(-1)

            # Apply mask to physical network
            valid_physical_indices = pei[pem]
            valid_physical_weights = pew[pem]

            # Apply mask to spanning network
            valid_spanning_indices = sei[sem]
            valid_spanning_weights = sew[sem]

            # Transpose indices to shape [2, E]
            valid_physical_indices = valid_physical_indices.t()
            valid_spanning_indices = valid_spanning_indices.t()

            # Add self loops to gather node specific features
            valid_physical_indices, valid_physical_weights = add_self_loops(
                valid_physical_indices, valid_physical_weights, fill_value=1.0, num_nodes=num_nodes)

            # Add self loops to gather node specific features
            valid_spanning_indices, valid_spanning_weights = add_self_loops(
                valid_spanning_indices, valid_spanning_weights, fill_value=1.0, num_nodes=num_nodes)

            # GCN Layer
            x_full = self.gcn_full(nf, valid_physical_indices, valid_physical_weights)
            x_mst = self.gcn_mst(nf, valid_spanning_indices, valid_spanning_weights)

            # GCN Activation
            x_full = F.relu(x_full)
            x_mst = F.relu(x_mst)

            # GCN Aggregator
            x_full_pooled = gnn.global_mean_pool(x_full, batch=torch.zeros(num_nodes, dtype=torch.long, device=self.device))
            x_mst_pooled = gnn.global_mean_pool(x_mst, batch=torch.zeros(num_nodes, dtype=torch.long, device=self.device))

            # Combine features from both GCN outputs
            combined_features = torch.cat([x_full_pooled, x_mst_pooled], dim=-1)
            combined_features_list.append(combined_features)

        # Concatenate pooled features from all graphs in the batch
        final_combined_features = torch.cat(combined_features_list, dim=0)

        return final_combined_features

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

        # Action dimensions based on the action space (assumes MultiDiscrete space)
        action_dims = action_space.nvec.tolist()

        # # Constructing the actor network.
        # # It needs to handle different actions for each part of the MultiDiscrete space.
        # # We use ModuleList to handle each discrete action dimension separately.
        # self.actor_modules = nn.ModuleList()
        # for dim in action_dims:
        #     # Each dimension of the action space has its own linear layer followed by softmax
        #     # This outputs a probability distribution over possible actions for each part.
        #     action_module = nn.Sequential(
        #         nn.Linear(features_dim, dim),
        #         nn.Softmax(dim=-1)
        #     )
        #     self.actor_modules.append(action_module.to(device))

        # # Constructing the critic network.
        # # It outputs a single value representing the state value estimate.
        # self.critic = nn.Sequential(
        #     nn.Linear(features_dim, 1)
        # ).to(device)

        # Constructing a more complex actor network.
        self.actor_modules = nn.ModuleList()
        for dim in action_dims:
            # Adding more layers with nonlinearities and possibly dropout for regularization.
            action_module = nn.Sequential(
                nn.Linear(features_dim, features_dim * 2),
                nn.ReLU(),
                nn.Dropout(p=0.2),  # Dropout for regularization
                nn.Linear(features_dim * 2, features_dim),  # Additional intermediate layer
                nn.ReLU(),
                nn.Linear(features_dim, dim),
                nn.Softmax(dim=-1)
            )
            self.actor_modules.append(action_module.to(device))

        # Constructing a more complex critic network.
        self.critic = nn.Sequential(
            nn.Linear(features_dim, features_dim * 2),
            nn.ReLU(),
            nn.Linear(features_dim * 2, features_dim),
            nn.ReLU(),
            nn.Linear(features_dim, 1)
        ).to(device)

        # Assigning the actor network list to action_net, used by the base class.
        self.action_net = self.actor_modules
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
        features = self.features_extractor(obs)
        return self.value_net(features)
    
    def forward(self, obs, deterministic=False):

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

        # Initialize lists to store actions and their log probabilities for each part of the action space
        actions = []
        action_log_probs = []

        # Loop through each module in the actor network (each corresponds to a part of the MultiDiscrete action space)
        for actor_module in self.action_net:
            # Obtain action probabilities from the current module
            action_probs = actor_module(features)
            # Create a categorical distribution based on the action probabilities
            dist = torch.distributions.Categorical(action_probs)
            # Sample or select the maximum probability action depending on whether deterministic is set
            action = dist.sample() if not deterministic else torch.argmax(action_probs, dim=1, keepdim=True)
            # Compute the log probability of the selected action
            log_prob = dist.log_prob(action)

            # Append the results to their respective lists
            actions.append(action)
            action_log_probs.append(log_prob)

        # Concatenate actions and log probabilities across all action dimensions
        actions = torch.cat(actions, dim=0).unsqueeze(0)
        action_log_probs = torch.cat(action_log_probs, dim=0).sum(dim=0, keepdim=True).unsqueeze(0)

        # Evaluate the state value using the critic network
        values = self.critic(features)

        # Return the actions, their log probabilities, and the state values
        return actions, values, action_log_probs

   
    def evaluate_actions(self, obs, actions):
        # Extract features from observations using the feature extractor
        features = self.features_extractor(obs)
        # Evaluate the state value using the critic network
        values = self.critic(features)

        # Lists to hold log probabilities and entropies for each action part
        action_log_probs = []
        entropies = []

        # Loop through each actor module and corresponding action dimension
        for i, actor_module in enumerate(self.action_net):
            # Get action probabilities from the actor module
            action_probs = actor_module(features)
            # Create a categorical distribution with the obtained probabilities
            dist = torch.distributions.Categorical(action_probs)
            # Calculate the log probability for the given action
            log_prob = dist.log_prob(actions[:, i])
            # Calculate the entropy of the distribution, a measure of randomness
            entropy = dist.entropy()

            # Append the log probability and entropy to their lists
            action_log_probs.append(log_prob)
            entropies.append(entropy)

        # Sum log probabilities and average entropies across all action dimensions
        action_log_probs = torch.stack(action_log_probs).sum(dim=0).unsqueeze(1)
        entropies = torch.stack(entropies).mean(dim=0).unsqueeze(1)

        # Return the state values, log probabilities of the actions, and entropies
        return values, action_log_probs, entropies

    def get_distribution(self, obs):
        """Get the action distribution based on the policy network's output."""
        # Extract features from the observation
        features = self.features_extractor(obs)
        distributions = []
        # For each actor module, get the action probabilities and form a categorical distribution
        for actor_module in self.action_net:
            action_probs = actor_module(features)
            distributions.append(torch.distributions.Categorical(action_probs))
        # Return a list of categorical distributions, one for each action part
        return distributions

    def _predict(self, obs, deterministic=False):
        """Predict actions based on the policy distribution and whether to use deterministic actions."""
        # Get the distributions for each action part
        distributions = self.get_distribution(obs)
        if deterministic:
            # If deterministic, choose the action with the highest probability
            actions = [torch.argmax(d.probs, dim=1) for d in distributions]
        else:
            # Otherwise, sample from the distribution
            actions = [d.sample() for d in distributions]
        # Stack the actions from all parts into a single tensor
        actions = torch.cat(actions, dim=0)
        return actions

