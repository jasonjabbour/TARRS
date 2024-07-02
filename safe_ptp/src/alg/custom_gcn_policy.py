import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

class GNNFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super(GNNFeatureExtractor, self).__init__(observation_space, features_dim)

        # Define the number of features each node has - update this based on actual features used
        node_feature_dim = 2  # Example: just using the MST and attacked status here
        self.gnn = gnn.GCNConv(node_feature_dim, features_dim)

    def forward(self, observations):
        # Extract relevant matrices and vectors
        full_network = observations['full_network']
        mst_status = observations['mst']
        attacked_status = observations['attacked']

        # Creating edge_index from full_network adjacency matrix
        edge_index = (full_network > 0).nonzero(as_tuple=False).t().contiguous()

        # Combining node features
        x = torch.cat([mst_status.unsqueeze(-1), attacked_status.unsqueeze(-1)], dim=-1)

        # Processing graph data through GNN
        x = self.gnn(x.float(), edge_index)
        x = F.relu(x)
        x = gnn.global_mean_pool(x, batch=torch.zeros(x.size(0), dtype=torch.long))  # Simplified pooling

        return x

class CustomGNNActorCriticPolicy(ActorCriticPolicy):
    def _build(self, lr_schedule):
        super()._build(lr_schedule)
        
        # Override the feature extractor with our custom GNNFeatureExtractor
        self.features_extractor = GNNFeatureExtractor(self.observation_space, features_dim=256)
        
        # Redefine the actor and critic networks
        self.actor = nn.Sequential(
            nn.Linear(256, self.action_space.n),  # Assuming action_space.n gives the number of possible edges
            nn.Sigmoid()
        )
        self.critic = nn.Sequential(
            nn.Linear(256, 1)
        )

    def forward(self, obs, deterministic: bool = False):
        features = self.extract_features(obs)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value, {}


