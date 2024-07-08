import torch
from torch import nn
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
        self.count = 0 

    def forward(self, observations):
        self.count+=1

        # print(self.count)
        # Extract relevant matrices and vectors
        full_network = observations['full_network'][0].to(self.device)
        mst_status = observations['mst'][0].to(self.device)
        attacked_status = observations['attacked'][0].to(self.device)
        weights = observations['weights'][0].to(self.device)
        
        # print("WEIGHTS!!!")
        # print(weights)
        # print(weights.shape)

        # Creating edge indices from adjacency matrices
        edge_index_full = (full_network > 0).nonzero(as_tuple=False).t().contiguous()
        edge_weights_full = weights[full_network > 0].clone().detach().float()

        edge_index_mst = (mst_status > 0).nonzero(as_tuple=False).t().contiguous()
        edge_weights_mst = weights[mst_status > 0].clone().detach().float()

        # print("Full Network")
        # print(full_network)
        # print(full_network.shape)
        # print("MST Status")
        # print(mst_status)
        # print(mst_status.shape)
        # print("Attacked Status")
        # print(attacked_status)
        # print(attacked_status.shape)
        # print("Edge Index FUll")
        # print(edge_index_full)
        # print(edge_index_full.shape)
        # print("Edge Index MST")
        # print(edge_index_mst)
        # print(edge_index_mst.shape)

        # print("Edge weights full")
        # print(edge_weights_full)
        # print(edge_weights_full.shape)

        # print("Edge weights mst")
        # print(edge_weights_mst)
        # print(edge_weights_mst.shape)

        # Prepare node features: here we use only the attacked status as node features for simplicity
        node_features = attacked_status.unsqueeze(-1).float()  # Assuming attacked_status is a [N,] vector

        # print("Node Features", node_features)
        # print(node_features.shape)

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
        super(CustomGNNActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=None,
            **kwargs
        )
        self.features_extractor = GNNFeatureExtractor(observation_space, features_dim).to(self.device)
        
        # Define the actor and critic networks using the extracted features
        self.actor = nn.Sequential(
            nn.Linear(features_dim, action_space.n),
            nn.Sigmoid()
        ).to(self.device)
        self.critic = nn.Sequential(
            nn.Linear(features_dim, 1)
        ).to(self.device)

    def forward(self, obs, deterministic=False):

        assert all(key in obs for key in ['full_network', 'mst', 'attacked', 'weights']), "Missing keys in observations"
        # Ensure all entries are tensors or compatible with tensor operations
        for key, value in obs.items():
            assert torch.is_tensor(value), f"Non-tensor data found in observations for key {key}"


        # Check and convert inputs
        if isinstance(obs, tuple):
            # If the observation is a tuple, unpack it as needed
            obs = obs[0]  

        # Ensure that each item in the observation dictionary is converted to a tensor
        obs_tensors = {k: v.to(self.device) if torch.is_tensor(v) else torch.tensor(v, dtype=torch.float32).to(self.device) for k, v in obs.items()}

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

        # print(type(actions))
        # print(type(values))
        # print(type(log_probs))
        return actions, values, log_probs






