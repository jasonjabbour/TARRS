import torch
from torch import nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy


class GNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(GNNFeatureExtractor, self).__init__(observation_space, features_dim)

        # Define the number of features each node has
        # initially just the attacked status
        node_feature_dim = 1
        self.gcn_full = gnn.GCNConv(node_feature_dim, features_dim // 2)
        self.gcn_mst = gnn.GCNConv(node_feature_dim, features_dim // 2)

    def forward(self, observations):
        # Extract relevant matrices and vectors
        full_network = observations['full_network']
        mst_status = observations['mst']
        attacked_status = observations['attacked']

        # Creating edge indices from adjacency matrices
        edge_index_full = (full_network > 0).nonzero(as_tuple=False).t().contiguous()
        edge_index_mst = (mst_status > 0).nonzero(as_tuple=False).t().contiguous()

        print("Full Network")
        print(full_network)
        print("MST Status")
        print(mst_status)
        print("Attacked Status")
        print(attacked_status)

        print("Edge Index FUll")
        print(edge_index_full)
        print("Edge Index MST")
        print(edge_index_mst)

        # Prepare node features: here we use only the attacked status as node features for simplicity
        node_features = attacked_status.unsqueeze(-1).float()  # Assuming attacked_status is a [N,] vector

        print("Node Features", node_features)
        # Processing graph data through GCNs
        x_full = self.gcn_full(node_features, edge_index_full)
        x_mst = self.gcn_mst(node_features, edge_index_mst)

        # Applying non-linearity (ReLU)
        x_full = F.relu(x_full)
        x_mst = F.relu(x_mst)

        # Pooling the features from both GCNs
        x_full_pooled = gnn.global_mean_pool(x_full, batch=torch.zeros(x_full.size(0), dtype=torch.long))
        x_mst_pooled = gnn.global_mean_pool(x_mst, batch=torch.zeros(x_mst.size(0), dtype=torch.long))

        # Concatenate features from both GCN outputs
        combined_features = torch.cat([x_full_pooled, x_mst_pooled], dim=-1)

        return combined_features

class CustomGNNActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, features_dim=256, **kwargs):
        super(CustomGNNActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=None,
            **kwargs
        )
        self.features_extractor = GNNFeatureExtractor(observation_space, features_dim)
        
        # Define the actor and critic networks using the extracted features
        self.actor = nn.Sequential(
            nn.Linear(features_dim, action_space.n),
            nn.Sigmoid()
        )
        self.critic = nn.Sequential(
            nn.Linear(features_dim, 1)
        )

    def forward(self, obs, deterministic=False):
        # Check and convert inputs
        if isinstance(obs, tuple):
            # If the observation is a tuple, unpack it as needed
            obs = obs[0]  

        # Ensure that each item in the observation dictionary is converted to a tensor
        obs_tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in obs.items()}

        # Extract features using the custom feature extractor
        features = self.features_extractor(obs_tensors)  # Directly use the feature extractor here

        # Generate actions and values from the neural networks
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value, {}






