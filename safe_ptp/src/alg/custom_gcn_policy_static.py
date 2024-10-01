"GCN Policy with only the spanning tree nework. Assuming physical network does not change."


import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Bernoulli
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.utils import add_self_loops, degree
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class GNNFeatureExtractorStatic(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(GNNFeatureExtractorStatic, self).__init__(observation_space, features_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the number of features each node has
        node_feature_dim = observation_space['physical_node_features'].shape[1]
        
        # GIN Layers with MLPs for Spanning Tree
        self.gin_mst1 = gnn.GINConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(node_feature_dim, features_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(features_dim, features_dim)
            )
        ).to(self.device)
        
        self.gin_mst2 = gnn.GINConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(features_dim, features_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(features_dim, features_dim)
            )
        ).to(self.device)

        self.gin_mst3 = gnn.GINConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(features_dim, features_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(features_dim, features_dim)
            )
        ).to(self.device)

    def forward(self, observations):
        # Keep batch dimensions intact, move to device and cast types
        spanning_node_features = observations['spanning_node_features'].to(self.device).float()

        spanning_edge_indices = observations['spanning_tree_edge_indices'].to(self.device).long()
        spanning_edge_weights = observations['spanning_tree_edge_weights'].to(self.device).float()
        spanning_edge_mask = observations['spanning_tree_edge_mask'].to(self.device).bool()

        features_list = []
        features_unflattened_list = []
    
        for idx in range(spanning_node_features.size(0)):
            nf_spanning = spanning_node_features[idx] 
            num_nodes = nf_spanning.size(0)
            
            # Spanning Tree Network Processing
            sei = spanning_edge_indices[idx]
            sew = spanning_edge_weights[idx]
            sem = spanning_edge_mask[idx].squeeze(-1)

            # Apply mask to spanning network
            valid_spanning_indices = sei[sem]
            valid_spanning_weights = sew[sem]

            # Transpose indices to shape [2, E]
            valid_spanning_indices = valid_spanning_indices.t()

            # Ensure the graph is undirected by adding reverse edges
            spanning_src, spanning_dst = valid_spanning_indices

            undirected_valid_spanning_indices = torch.cat([valid_spanning_indices, torch.stack([spanning_dst, spanning_src], dim=0)], dim=1)
            undirected_valid_spanning_weights = torch.cat([valid_spanning_weights, valid_spanning_weights], dim=0)

            # Add self loops to gather node specific features
            undirected_valid_spanning_indices, undirected_valid_spanning_weights = add_self_loops(
                undirected_valid_spanning_indices, undirected_valid_spanning_weights, fill_value=1.0, num_nodes=num_nodes)

            # Color Refinement: Apply multiple GIN Layers with ReLU activation for spanning tree network
            x_mst = self.gin_mst1(nf_spanning, undirected_valid_spanning_indices)
            x_mst = F.relu(x_mst)
            x_mst = self.gin_mst2(x_mst, undirected_valid_spanning_indices)
            x_mst = F.relu(x_mst)
            x_mst = self.gin_mst3(x_mst, undirected_valid_spanning_indices)
            x_mst = F.relu(x_mst)

            # Store the original (non-flattened) features (for easy access later) [num nodes, embedding features]
            features_unflattened_list.append(x_mst)

            # Flatten the GIN outputs
            x_mst_flat = x_mst.flatten(start_dim=0)

            # Combine features from both GIN outputs [numnodes * embedding features]
            features_list.append(x_mst_flat)

        # Unflattened features from all graphs in batch [batch, num nodes, embedding features]
        final_features_unflattened = torch.stack(features_unflattened_list, dim=0)

        # Concatenate features from all graphs in the batch [batch, num nodes * embedding features]
        final_features = torch.stack(features_list, dim=0)


        return final_features, final_features_unflattened


class CustomGNNActorCriticPolicyStatic(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, features_dim=32, **kwargs):
        # Call the super constructor first
        super(CustomGNNActorCriticPolicyStatic, self).__init__(
            observation_space, action_space, lr_schedule,
            features_extractor_class=GNNFeatureExtractorStatic,
            features_extractor_kwargs={'features_dim': features_dim},
            **kwargs
        )

        # Get the device dynamically based on CUDA availability
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # This is the number of nodes
        self.num_nodes = observation_space['spanning_node_features'].shape[0]  
        # Max number of edges in a fully connected graph
        self.num_edges = int(self.num_nodes * (self.num_nodes - 1) / 2)  

        self.features_dim = features_dim

        # Define the first actor network for selecting the first node
        self.actor = nn.Sequential(
            nn.Linear(self.features_dim * self.num_nodes, (self.features_dim * self.num_nodes)//2),
            nn.ReLU(),
            nn.Linear((self.features_dim * self.num_nodes)//2, self.num_edges),
            nn.ReLU()
        ).to(self._device)

        # Define the critic network for evaluating the state value
        self.critic = nn.Sequential(
            nn.Linear(self.features_dim * self.num_nodes, (self.features_dim * self.num_nodes)//2),
            nn.ReLU(),
            nn.Linear((self.features_dim * self.num_nodes)//2, (self.features_dim * self.num_nodes)//2),
            nn.ReLU(),
            nn.Linear((self.features_dim * self.num_nodes)//2, 1)
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
        features, _ = self.features_extractor(obs_tensors)  
        # Execute model prediction
        return self.value_net(features) 
    
    def forward(self, obs, deterministic=False):

        # Check and convert inputs
        if isinstance(obs, tuple):
            # If the observation is a tuple, unpack it as needed
            obs = obs[0]

        # Convert all observation inputs to tensors and ensure they are on the correct device
        obs_tensors = {k: torch.as_tensor(v, dtype=torch.float32).to(self._device) if not torch.is_tensor(v) else v.to(self._device) for k, v in obs.items()}

        # Extract features from observations using the custom feature extractor
        features, features_unflattened = self.features_extractor(obs_tensors)

        # Compute logits using the actor network (this now outputs logits for each edge)
        logits = self.actor(features)

        # Apply sigmoid to get probabilities of edge presence
        probabilities = torch.sigmoid(logits)

        # Create a Bernoulli distribution for each edge based on the probabilities
        dist = Bernoulli(probabilities)

        # Sample or select the single maximum probability action based on whether deterministic mode is on
        if deterministic:
            action = (probabilities > 0.5).float()  # Deterministic threshold
        else:
            action = dist.sample()  # Sample from the Bernoulli distribution

        # Evaluate the state value using the critic network
        values = self.critic(features)

        # Return the action, its log probability, and the state values
        log_prob = dist.log_prob(action).sum(dim=-1)  # Sum log probabilities across edges
        return action, values, log_prob

   
    def evaluate_actions(self, obs, actions):

        # Convert all observation inputs to tensors and ensure they are on the correct device
        obs_tensors = {k: torch.as_tensor(v, dtype=torch.float32).to(self._device) if not torch.is_tensor(v) else v.to(self._device) for k, v in obs.items()}

        # Extract features from observations using the custom feature extractor
        features, features_unflattened = self.features_extractor(obs_tensors)

        # Compute logits using the actor network
        logits = self.actor(features)

        # Apply sigmoid to get probabilities for edge presence
        probabilities = torch.sigmoid(logits)

        # Create a Bernoulli distribution for each edge based on the probabilities
        dist = Bernoulli(probabilities)

        # Calculate the log probability of the given actions
        log_prob = dist.log_prob(actions).sum(dim=-1)  # Sum log probabilities across edges

        # Calculate the entropy of the distribution
        entropy = dist.entropy().sum(dim=-1)  # Sum entropy across edges

        # Evaluate the state value using the critic network
        values = self.critic(features)

        # Return the state values, log probabilities of the actions, and entropy
        return values, log_prob.view(-1, 1), entropy.view(-1, 1)


    def get_distribution(self, obs):
        """Get the action distribution based on the policy network's output."""

        # Convert all observation inputs to tensors and ensure they are on the correct device
        obs_tensors = {k: torch.as_tensor(v, dtype=torch.float32).to(self._device) if not torch.is_tensor(v) else v.to(self._device) for k, v in obs.items()}
        
        # Extract features and action masks from the observation
        features, features_unflattened = self.features_extractor(obs_tensors)

        # Compute logits using the actor network (for all edges)
        logits = self.actor(features)

        # Apply sigmoid to convert logits to probabilities (between 0 and 1 for each edge)
        probabilities = torch.sigmoid(logits)

        # Create a Bernoulli distribution for each edge based on the probabilities
        dist = Bernoulli(probabilities)

        return dist, features


    def _predict(self, obs, deterministic=False):
        """Predict actions based on the policy distribution and whether to use deterministic actions."""

        # Get the action distribution and features
        dist, features = self.get_distribution(obs)

        if deterministic:
            # If deterministic, choose 1 for probabilities > 0.5, otherwise 0
            action = (dist.probs > 0.5).float()
        else:
            # Otherwise, sample from the Bernoulli distribution
            action = dist.sample()

        return action