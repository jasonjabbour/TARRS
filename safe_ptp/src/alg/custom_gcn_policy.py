import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Bernoulli
import torch.nn.functional as F
import torch_geometric.nn as gnn
from torch_geometric.utils import add_self_loops, degree
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class GNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(GNNFeatureExtractor, self).__init__(observation_space, features_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the number of features each node has
        node_feature_dim = observation_space['physical_node_features'].shape[1]

        # GIN Layers with MLPs for Physical Full Network
        self.gin_full1 = gnn.GINConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(node_feature_dim, features_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(features_dim // 2, features_dim // 2)
            )
        ).to(self.device)
        
        self.gin_full2 = gnn.GINConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(features_dim // 2, features_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(features_dim // 2, features_dim // 2)
            )
        ).to(self.device)
    
        self.gin_full3 = gnn.GINConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(features_dim // 2, features_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(features_dim // 2, features_dim // 2)
            )
        ).to(self.device)
        
        # GIN Layers with MLPs for Spanning Tree
        self.gin_mst1 = gnn.GINConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(node_feature_dim, features_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(features_dim // 2, features_dim // 2)
            )
        ).to(self.device)
        
        self.gin_mst2 = gnn.GINConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(features_dim // 2, features_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(features_dim // 2, features_dim // 2)
            )
        ).to(self.device)

        self.gin_mst3 = gnn.GINConv(
            nn=torch.nn.Sequential(
                torch.nn.Linear(features_dim // 2, features_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(features_dim // 2, features_dim // 2)
            )
        ).to(self.device)

    def forward(self, observations):
        # Keep batch dimensions intact, move to device and cast types
        physical_node_features = observations['physical_node_features'].to(self.device).float()
        spanning_node_features = observations['spanning_node_features'].to(self.device).float()

        physical_edge_indices = observations['physical_edge_indices'].to(self.device).long()
        physical_edge_weights = observations['physical_edge_weights'].to(self.device).float()
        physical_edge_mask = observations['physical_edge_mask'].to(self.device).bool()

        spanning_edge_indices = observations['spanning_tree_edge_indices'].to(self.device).long()
        spanning_edge_weights = observations['spanning_tree_edge_weights'].to(self.device).float()
        spanning_edge_mask = observations['spanning_tree_edge_mask'].to(self.device).bool()

        features_list = []
        features_unflattened_list = []
    
        for idx in range(physical_node_features.size(0)):
            nf_physical = physical_node_features[idx]  # Node features for one graph instance
            nf_spanning = spanning_node_features[idx] 
            num_nodes = nf_physical.size(0)
            
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

            # Ensure the graph is undirected by adding reverse edges
            physical_src, physical_dst = valid_physical_indices
            spanning_src, spanning_dst = valid_spanning_indices

            undirected_valid_physical_indices = torch.cat([valid_physical_indices, torch.stack([physical_dst, physical_src], dim=0)], dim=1)
            undirected_valid_physical_weights = torch.cat([valid_physical_weights, valid_physical_weights], dim=0)

            undirected_valid_spanning_indices = torch.cat([valid_spanning_indices, torch.stack([spanning_dst, spanning_src], dim=0)], dim=1)
            undirected_valid_spanning_weights = torch.cat([valid_spanning_weights, valid_spanning_weights], dim=0)

            # Add self loops to gather node specific features
            undirected_valid_physical_indices, undirected_valid_physical_weights = add_self_loops(
                undirected_valid_physical_indices, undirected_valid_physical_weights, fill_value=1.0, num_nodes=num_nodes)

            # Add self loops to gather node specific features
            undirected_valid_spanning_indices, undirected_valid_spanning_weights = add_self_loops(
                undirected_valid_spanning_indices, undirected_valid_spanning_weights, fill_value=1.0, num_nodes=num_nodes)

            # Color Refinement: Apply multiple GIN Layers with ReLU activation for physical network
            x_full = self.gin_full1(nf_physical, undirected_valid_physical_indices)
            x_full = F.relu(x_full)
            x_full = self.gin_full2(x_full, undirected_valid_physical_indices)
            x_full = F.relu(x_full)
            x_full = self.gin_full3(x_full, undirected_valid_physical_indices)
            x_full = F.relu(x_full)

            # Color Refinement: Apply multiple GIN Layers with ReLU activation for spanning tree network
            x_mst = self.gin_mst1(nf_spanning, undirected_valid_spanning_indices)
            x_mst = F.relu(x_mst)
            x_mst = self.gin_mst2(x_mst, undirected_valid_spanning_indices)
            x_mst = F.relu(x_mst)
            x_mst = self.gin_mst3(x_mst, undirected_valid_spanning_indices)
            x_mst = F.relu(x_mst)

            # Store the original (non-flattened) features (for easy access later) [num nodes, embedding features]
            features_unflattened = torch.cat([x_full, x_mst], dim=-1)
            features_unflattened_list.append(features_unflattened)

            # Flatten the GIN outputs
            x_full_flat = x_full.flatten(start_dim=0)
            x_mst_flat = x_mst.flatten(start_dim=0)

            # Combine features from both GIN outputs [numnodes * embedding features]
            combined_features = torch.cat([x_full_flat, x_mst_flat], dim=-1)
            features_list.append(combined_features)

        # Unflattened features from all graphs in batch [batch, num nodes, embedding features]
        final_features_unflattened = torch.stack(features_unflattened_list, dim=0)

        # Concatenate features from all graphs in the batch [batch, num nodes * embedding features]
        final_features = torch.stack(features_list, dim=0)


        return final_features, final_features_unflattened


class CustomGNNActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, features_dim=32, **kwargs):
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

        self.features_dim = features_dim

        # Define the first actor network for selecting the first node
        self.actor_first = nn.Sequential(
            nn.Linear(self.features_dim * self.num_nodes, (self.features_dim * self.num_nodes)//2),
            nn.ReLU(),
            nn.Linear((self.features_dim * self.num_nodes)//2, self.num_nodes),
            nn.ReLU()
        ).to(self._device)

        # Define the second actor network for selecting the second node based on the first node's embedding
        self.actor_second = nn.Sequential(
            nn.Linear((self.features_dim * self.num_nodes) + self.features_dim, (self.features_dim * self.num_nodes)//2),
            nn.ReLU(),
            nn.Linear((self.features_dim * self.num_nodes)//2, self.num_nodes),
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
        self.action_net = nn.ModuleList([self.actor_first, self.actor_second])
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

        # Extract the action mask from the observations, ensuring it's in the correct format and device
        action_mask = obs_tensors['action_mask'].to(self._device)

        # Get the mask for the first node selection
        first_node_action_mask = obs_tensors['first_node_action_mask'].to(self._device)

        # Compute logits using the first actor network
        logits_first = self.actor_first(features)

        # TODO: CHECK IF MASK IS CORRECT
        # Apply the spanning tree mask to ensure the first node is from the spanning tree (or any node if the spanning tree is empty)
        # Also make sure that each node has a potential new edge it can connect to
        if first_node_action_mask.sum() > 0:
            # Combine the first node action mask with the action mask to ensure the first node has potential edges
            combined_mask = first_node_action_mask * (action_mask.sum(dim=-1) > 0).float()
            # Invalid actions get set to '-inf' to exclude them from selection
            masked_logits_first = torch.where(combined_mask.bool(), logits_first, torch.tensor(float('-inf')).to(self._device))
        else:
            masked_logits_first = logits_first

        # Apply softmax to convert masked logits into probabilities and view as a flat vector
        probabilities_first = F.softmax(masked_logits_first.flatten(1), dim=-1)

        # Create a categorical distribution from the probabilities to sample actions
        dist_first = torch.distributions.Categorical(probabilities_first)

        # Sample or select the single maximum probability action based on whether deterministic mode is on
        first_node = dist_first.sample() if not deterministic else torch.argmax(probabilities_first, dim=-1)

        # Fetch the features of the first selected node from unflattened features
        first_node_features = torch.stack([features_unflattened[i, first_node[i], :] for i in range(first_node.size(0))], dim=0)

        # Combine the features of the first node with each node's features
        combined_features = torch.cat([features, first_node_features], dim=-1)

        # Compute logits using the second actor network
        logits_second = self.actor_second(combined_features).view(-1, self.num_nodes)

        # TODO: CHECK IF MASK IS CORRECT
        # Apply the action mask for the second node selection
        masked_logits_second = torch.where(action_mask[torch.arange(action_mask.size(0)), first_node], logits_second, torch.tensor(float('-inf')).to(self._device))

        # Apply softmax to convert masked logits into probabilities for the second node
        probabilities_second = F.softmax(masked_logits_second.flatten(1), dim=-1)

        # Create a categorical distribution from the probabilities to sample actions
        dist_second = torch.distributions.Categorical(probabilities_second)

        # Sample or select the single maximum probability action based on whether deterministic mode is on
        second_node = dist_second.sample() if not deterministic else torch.argmax(probabilities_second, dim=-1)

        # Combine the selected nodes into an action tensor
        action = torch.stack([first_node, second_node], dim=-1)

        # Compute the log probabilities for the selected nodes
        log_prob_first = dist_first.log_prob(first_node).view(-1, 1)
        log_prob_second = dist_second.log_prob(second_node).view(-1, 1)
        log_prob = log_prob_first + log_prob_second
 
        # Evaluate the state value using the critic network
        values = self.critic(features)

        # Return the actions, their log probabilities, and the state values
        return action, values, log_prob
   
    def evaluate_actions(self, obs, actions):

        # Convert all observation inputs to tensors and ensure they are on the correct device
        obs_tensors = {k: torch.as_tensor(v, dtype=torch.float32).to(self._device) if not torch.is_tensor(v) else v.to(self._device) for k, v in obs.items()}

        # Extract features from observations using the custom feature extractor
        features, features_unflattened = self.features_extractor(obs_tensors)

        # Extract the action mask from the observations, ensuring it's in the correct format and device
        action_mask = obs_tensors['action_mask'].to(self._device)

        # Get the mask for the first node selection
        first_node_action_mask = obs_tensors['first_node_action_mask'].to(self._device)

        # Compute logits using the first actor network
        logits_first = self.actor_first(features)
        
        # Apply the spanning tree mask to ensure the first node is from the spanning tree (or any node if the spanning tree is empty)
        if first_node_action_mask.sum() > 0:
            # Combine the first node action mask with the action mask to ensure the first node has potential edges
            combined_mask = first_node_action_mask * (action_mask.sum(dim=-1) > 0).float()
            # Invalid actions get set to '-inf' to exclude them from selection
            masked_logits_first = torch.where(combined_mask.bool(), logits_first, torch.tensor(float('-inf')).to(self._device))
        else:
            masked_logits_first = logits_first

        # Apply softmax to convert masked logits into probabilities and view as a flat vector
        probabilities_first = F.softmax(masked_logits_first.flatten(1), dim=-1)

        # Create a categorical distribution from the probabilities to sample actions
        dist_first = torch.distributions.Categorical(probabilities_first)

        # Extract the first node from actions
        first_node = actions[:, 0]

        # Fetch the features of the first selected node from unflattened features
        first_node_features = torch.stack([features_unflattened[i, first_node[i], :] for i in range(first_node.size(0))], dim=0)

        # Combine the features of the first node with each node's features
        combined_features = torch.cat([features, first_node_features], dim=-1)

        # Compute logits using the second actor network
        logits_second = self.actor_second(combined_features).view(-1, self.num_nodes)

        # Apply the action mask for the second node selection
        masked_logits_second = torch.where(action_mask[torch.arange(action_mask.size(0)), first_node], logits_second, torch.tensor(float('-inf')).to(self._device))

        # Apply softmax to convert masked logits into probabilities for the second node
        probabilities_second = F.softmax(masked_logits_second.flatten(1), dim=-1)

        # Create a categorical distribution from the probabilities
        dist_second = torch.distributions.Categorical(probabilities_second)

        # Extract the second node from actions
        second_node = actions[:, 1]

        # Calculate the log probability for the given actions
        log_prob_first = dist_first.log_prob(first_node)
        log_prob_second = dist_second.log_prob(second_node)
        log_probs = log_prob_first + log_prob_second

        # Calculate the entropy of the distributions, a measure of randomness
        entropy = dist_first.entropy() + dist_second.entropy()

        # Evaluate the state value using the critic network
        values = self.critic(features)

        # Return the state values, log probabilities of the actions, and entropy
        return values, log_probs.view(-1, 1), entropy.view(-1, 1)


    def get_distribution(self, obs):
        """Get the action distribution based on the policy network's output."""

        # Convert all observation inputs to tensors and ensure they are on the correct device
        obs_tensors = {k: torch.as_tensor(v, dtype=torch.float32).to(self._device) if not torch.is_tensor(v) else v.to(self._device) for k, v in obs.items()}
        
        # Extract features and action masks from the observation
        features, features_unflattened = self.features_extractor(obs_tensors)
        action_mask = obs_tensors['action_mask'].to(self._device)
        first_node_action_mask = obs_tensors['first_node_action_mask'].to(self._device)

        # Compute logits using the first actor network
        logits_first = self.actor_first(features)

        # Apply the spanning tree mask and action mask to ensure the first node is valid and has potential edges
        if first_node_action_mask.sum() > 0:
            combined_mask = first_node_action_mask * (action_mask.sum(dim=-1) > 0).float()
            masked_logits_first = torch.where(combined_mask.bool(), logits_first, torch.tensor(float('-inf')).to(self._device))
        else:
            masked_logits_first = logits_first

        # Apply softmax to convert masked logits into probabilities and create a categorical distribution
        probabilities_first = F.softmax(masked_logits_first.flatten(1), dim=-1)
        dist_first = torch.distributions.Categorical(probabilities_first)

        return dist_first, features, features_unflattened, action_mask

    def _predict(self, obs, deterministic=False):
        """Predict actions based on the policy distribution and whether to use deterministic actions."""

        # Get the first stage distribution and features
        dist_first, features, features_unflattened, action_mask = self.get_distribution(obs)

        if deterministic:
            # If deterministic, choose the action with the highest probability
            first_node = torch.argmax(dist_first.probs, dim=1)
        else:
            # Otherwise, sample from the distribution
            first_node = dist_first.sample()

        # Fetch the features of the first selected node from unflattened features
        first_node_features = torch.stack([features_unflattened[i, first_node[i], :] for i in range(first_node.size(0))], dim=0)

        # Combine the features of the first node with each node's features
        combined_features = torch.cat([features, first_node_features], dim=-1)

        # Compute logits using the second actor network
        logits_second = self.actor_second(combined_features).view(-1, self.num_nodes)

        # Apply the action mask for the second node selection
        masked_logits_second = torch.where(action_mask[torch.arange(action_mask.size(0)), first_node], logits_second, torch.tensor(float('-inf')).to(self._device))

        # Apply softmax to convert masked logits into probabilities and create a categorical distribution
        probabilities_second = F.softmax(masked_logits_second.flatten(1), dim=-1)
        dist_second = torch.distributions.Categorical(probabilities_second)

        if deterministic:
            # If deterministic, choose the action with the highest probability
            second_node = torch.argmax(dist_second.probs, dim=1)
        else:
            # Otherwise, sample from the distribution
            second_node = dist_second.sample()

        # Combine the selected nodes into an action tensor
        action = torch.stack([first_node, second_node], dim=-1)

        return action