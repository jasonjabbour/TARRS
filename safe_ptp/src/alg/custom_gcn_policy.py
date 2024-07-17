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
        node_feature_dim = 1  # Includes attacked status
        self.gcn_full = gnn.GCNConv(node_feature_dim, features_dim).to(self.device)
        # self.gcn_mst = gnn.GCNConv(node_feature_dim, features_dim // 2).to(self.device)

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

        # spanning_edge_indices = observations['spanning_tree_edge_indices'].to(self.device).long()
        # spanning_edge_weights = observations['spanning_tree_edge_weights'].to(self.device).float()
        # spanning_edge_mask = observations['spanning_tree_edge_mask'].to(self.device).bool()

        features_list = []

        for idx in range(node_features.size(0)):
            nf = node_features[idx]  # Node features for one graph instance
            num_nodes = nf.size(0)

            # Physical Network Processing
            pei = physical_edge_indices[idx]
            pew = physical_edge_weights[idx]
            pem = physical_edge_mask[idx].squeeze(-1)

            # # Spanning Tree Network Processing
            # sei = spanning_edge_indices[idx]
            # sew = spanning_edge_weights[idx]
            # sem = spanning_edge_mask[idx].squeeze(-1)

            # Apply mask to physical network
            valid_physical_indices = pei[pem]
            valid_physical_weights = pew[pem]

            # # Apply mask to spanning network
            # valid_spanning_indices = sei[sem]
            # valid_spanning_weights = sew[sem]

            # Transpose indices to shape [2, E]
            valid_physical_indices = valid_physical_indices.t()
            # valid_spanning_indices = valid_spanning_indices.t()

            # Add self loops to gather node specific features
            valid_physical_indices, valid_physical_weights = add_self_loops(
                valid_physical_indices, valid_physical_weights, fill_value=1.0, num_nodes=num_nodes)

            # # Add self loops to gather node specific features
            # valid_spanning_indices, valid_spanning_weights = add_self_loops(
            #     valid_spanning_indices, valid_spanning_weights, fill_value=1.0, num_nodes=num_nodes)

            # GCN Layer
            x_full = self.gcn_full(nf, valid_physical_indices, valid_physical_weights)
            # x_mst = self.gcn_mst(nf, valid_spanning_indices, valid_spanning_weights)

            # GCN Activation
            x_full = F.relu(x_full)
            # x_mst = F.relu(x_mst)

            # GCN Aggregator
            x_full_pooled = gnn.global_mean_pool(x_full, batch=torch.zeros(num_nodes, dtype=torch.long, device=self.device))
            # x_mst_pooled = gnn.global_mean_pool(x_mst, batch=torch.zeros(num_nodes, dtype=torch.long, device=self.device))

            # Combine features from both GCN outputs
            # combined_features = torch.cat([x_full_pooled, x_mst_pooled], dim=-1)
            features_list.append(x_full_pooled)

        # Concatenate pooled features from all graphs in the batch
        final_combined_features = torch.cat(features_list, dim=0)

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
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Extract the size of the MultiDiscrete space, which equals the max number of nodes
        self.num_nodes = self.action_space.nvec[0]

        # Actor outputs logits for each node pair
        self.actor = nn.Sequential(
            nn.Linear(features_dim, features_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(features_dim * 2, self.num_nodes * self.num_nodes),
            nn.ReLU()
        ).to(self._device)

        self.critic = nn.Sequential(
            nn.Linear(features_dim, features_dim * 2),
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
        obs_tensors = {k: v.to(self._device) if torch.is_tensor(v) else torch.tensor(v, dtype=torch.float32).to(self._device) for k, v in obs.items()}
        # Ensure input tensors are on the right device
        features = self.features_extractor(obs_tensors)  
        # Execute model prediction
        return self.value_net(features) 
    
    def forward(self, obs, deterministic=False):

        # Check and convert inputs
        if isinstance(obs, tuple):
            # If the observation is a tuple, unpack it as needed
            obs = obs[0]

        # Ensure that each item in the observation dictionary is converted to a tensor and moved to the correct device
        obs_tensors = {k: v.to(self._device) if torch.is_tensor(v) else torch.tensor(v, dtype=torch.float32).to(self._device) for k, v in obs.items()}

        # Extract features using the custom feature extractor
        features = self.features_extractor(obs_tensors)  # Directly use the feature extractor here

        # Extract the action mask from the observations, ensuring it's in the correct format and device
        action_mask = obs_tensors['action_mask'].to(self._device)

        # Compute logits using the actor network and reshape them to match the action mask dimensions
        logits = self.actor(features).view(-1, self.num_nodes, self.num_nodes)

        # Apply the action mask to logits; invalid actions get set to '-inf' to exclude them from selection
        masked_logits = torch.where(action_mask.bool(), logits, torch.tensor(float('-inf')).to(self._device))

        # Apply softmax to convert masked logits into probabilities and view as a flat vector instead of a matrix
        probabilities = F.softmax(masked_logits.flatten(1), dim=-1) #.view(-1, self.num_nodes, self.num_nodes)

        # Create a categorical distribution from the probabilities to sample actions
        dist = torch.distributions.Categorical(probabilities)

        # Sample or select the single maximum probability action based on whether deterministic mode is on
        flat_action_index = dist.sample() if not deterministic else torch.argmax(probabilities, dim=1)

        # Convert flat index to 2D matrix indices representing the nodes
        node_i = flat_action_index // self.num_nodes
        node_j = flat_action_index % self.num_nodes

        # Create a tensor from the node indices
        action = torch.stack([node_i, node_j], dim=-1)

        # Fetch log probability handling any batch dimensions
        log_prob = dist.log_prob(flat_action_index).view(-1, 1)

        # Evaluate the state value using the critic network
        values = self.critic(features)

        # Return the actions, their log probabilities, and the state values
        return action, values, log_prob
   
    def evaluate_actions(self, obs, actions):

        # Convert all observation inputs to tensors and ensure they are on the correct device
        obs_tensors = {k: torch.as_tensor(v, dtype=torch.float32).to(self._device) if not torch.is_tensor(v) else v.to(self._device) for k, v in obs.items()}

        # Extract features from observations using the custom feature extractor
        features = self.features_extractor(obs_tensors)

        # Extract the action mask from the observations, ensuring it's in the correct format and device
        action_mask = obs_tensors['action_mask'].to(self._device)

        # Compute logits using the actor network and reshape them to match the action mask dimensions
        logits = self.actor(features).view(-1, self.num_nodes, self.num_nodes)

        # Apply the action mask to logits; invalid actions get set to '-inf' to exclude them from selection
        masked_logits = torch.where(action_mask.bool(), logits, torch.tensor(float('-inf')).to(self._device))

        # Apply softmax to convert masked logits into probabilities and view as a flat vector instead of a matrix
        probabilities = F.softmax(masked_logits.flatten(1), dim=-1)

        # Create a categorical distribution from the probabilities
        dist = torch.distributions.Categorical(probabilities)

        # Flatten the actions to match the probabilities shape for evaluation (do the opposite of the forward function)
        flat_actions = actions[:, 0] * self.num_nodes + actions[:, 1]

        # Calculate the log probability for the given actions
        log_probs = dist.log_prob(flat_actions)

        # Calculate the entropy of the distribution, a measure of randomness
        entropy = dist.entropy()

        # Evaluate the state value using the critic network
        values = self.critic(features)

        # Return the state values, log probabilities of the actions, and entropy
        return values, log_probs.view(-1, 1), entropy.view(-1, 1)


    def get_distribution(self, obs):
        """Get the action distribution based on the policy network's output."""

        obs_tensors = {k: torch.as_tensor(v, dtype=torch.float32).to(self._device) if not torch.is_tensor(v) else v.to(self._device) for k, v in obs.items()}
        
        # Extract features and action mask from the observation
        features = self.features_extractor(obs_tensors)
        action_mask = obs_tensors['action_mask'].to(self._device)
        
        # Compute logits and apply the action mask
        logits = self.actor(features).view(-1, self.num_nodes, self.num_nodes)
        masked_logits = torch.where(action_mask.bool(), logits, torch.tensor(float('-inf')).to(self._device))
        
        # Apply softmax to get probabilities
        probabilities = F.softmax(masked_logits, dim=-1)
        
        # Create a categorical distribution from these probabilities
        distribution = torch.distributions.Categorical(probabilities.view(-1, self.num_nodes * self.num_nodes))
        return distribution

    def _predict(self, obs, deterministic=False):
        """Predict actions based on the policy distribution and whether to use deterministic actions."""
        
        # Ensure the distribution is computed on the right device
        distribution = self.get_distribution(obs).to(self._device)
        
        if deterministic:
            # If deterministic, choose the action with the highest probability
            action_indices = torch.argmax(distribution.probs, dim=1)
        else:
            # Otherwise, sample from the distribution
            action_indices = distribution.sample()
        
        # Convert flat indices to matrix indices
        action_pairs = (action_indices // self.num_nodes, action_indices % self.num_nodes)
        return torch.stack(action_pairs, dim=1).to(self._device)  # Ensure that the output tensor is also on the correct device


    # def _predict(self, obs, deterministic=False):
    #     """Predict actions based on the policy distribution and whether to use deterministic actions."""
    #     # Get the distributions for each action part
    #     distributions = self.get_distribution(obs)
    #     if deterministic:
    #         # If deterministic, choose the action with the highest probability
    #         actions = [torch.argmax(d.probs, dim=1) for d in distributions]
    #     else:
    #         # Otherwise, sample from the distribution
    #         actions = [d.sample() for d in distributions]
    #     # Stack the actions from all parts into a single tensor
    #     actions = torch.cat(actions, dim=0)
    #     return actions



# # Initialize lists to store actions and their log probabilities for each part of the action space
# actions = []
# action_log_probs = []

# # Obtain action probabilities from the current module
# action_probs = actor_module(features)
# # Create a categorical distribution based on the action probabilities
# dist = torch.distributions.Categorical(action_probs)
# # Sample or select the maximum probability action depending on whether deterministic is set
# action = dist.sample() if not deterministic else torch.argmax(action_probs, dim=1, keepdim=True)
# # Compute the log probability of the selected action
# log_prob = dist.log_prob(action)

# # Append the results to their respective lists
# actions.append(action)
# action_log_probs.append(log_prob)

# # Concatenate actions and log probabilities across all action dimensions
# actions = torch.cat(actions, dim=0).unsqueeze(0)
# action_log_probs = torch.cat(action_log_probs, dim=0).sum(dim=0, keepdim=True).unsqueeze(0)
