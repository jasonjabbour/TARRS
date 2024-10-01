import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter as tk
import time
import collections

from safe_ptp.src.env.network_env import NetworkEnvironment

SHOW_WEIGHT_LABELS = False 
# Number of nodes that network increases by every difficulty level
NUM_NODE_INCREASE_RATE_PER_LEVEL = 1
# Number of nodes that can be attacked increased by every difficulty level
NUM_ATTACKED_NODE_INCREASE_RATE_PER_LEVEL = 1

class SpanningTreeEnv(gym.Env):
    def __init__(self, min_nodes, 
                       max_nodes, 
                       min_redundancy, 
                       start_difficulty_level=1,
                       final_difficulty_level=10,
                       num_timestep_cooldown=250000, 
                       min_attacked_nodes=1,
                       max_attacked_nodes=2,
                       show_weight_labels=False, 
                       render_mode=False, 
                       max_ep_steps=100, 
                       node_size=700, 
                       history_size=500, 
                       performance_threshold=4):
        super(SpanningTreeEnv, self).__init__()

        # Initialize parameters for the network environment
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.min_redundancy = min_redundancy
        self.min_attacked_nodes = min_attacked_nodes
        self.max_attacked_nodes = max_attacked_nodes

        # Curricula Parameters 
        self.current_level = start_difficulty_level # Initial Level
        self.performance_threshold = performance_threshold  # Define a suitable threshold for your task
        self.final_difficulty_level = final_difficulty_level # max difficulty level
        self.num_timestep_cooldown = num_timestep_cooldown # number of episodes before allowing level increase
        self.num_nodes_history = [] # number of nodes in a network tracking
        self.history_size = history_size # size of the deque for storing ep cummulative reward
        self.current_level_total_timesteps = 0
        
        # Set the level parameters to start
        self.update_level_parameters()

        # Deque with a fixed size to store reward history
        self.reward_history = collections.deque(maxlen=self.history_size)
        # Initialize cumulative reward tracker
        self.ep_cumulative_reward = 0  

        # Parameter to control weight label rendering
        self.show_weight_labels = show_weight_labels 

        # Permit Rendering
        self.render_mode = render_mode

        # Max Episode Steps
        self.max_ep_steps = max_ep_steps
        self.current_step = 0

        # Visualization parameters
        self.node_size = node_size

        # Initialize placeholders for the network environment and graphs
        self.network_env = None
        self.network = None
        self.tree = None
        self.action_mask = None
        self.first_node_action_mask = None
       
        # Initialize placeholders for the number of nodes, action space, and observation space
        self.max_difficulty_num_nodes = 4 + NUM_NODE_INCREASE_RATE_PER_LEVEL *  self.final_difficulty_level

        # number of possible edges in an undirected graph without self-loops
        self.max_difficulty_max_num_edges = int(self.max_difficulty_num_nodes * (self.max_difficulty_num_nodes - 1) / 2) 

        # # Flat array representing only the upper triangle of the adjacency matrix. (FOR PPO)
        # self.action_space = spaces.MultiBinary(self.max_difficulty_max_num_edges)

        # Define a continuous action space where each action can range from 0 to 1 (FOR SAC)
        # self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.max_difficulty_max_num_edges,), dtype=np.float32)

        # # Define the observation space 
        # # TODO currently observation space is max number of nodes. Explore embedding to an equal dimension.
        # self.observation_space = spaces.Dict({
        #     "physical_network": spaces.Box(low=0, high=1, shape=(self.max_difficulty_num_nodes, self.max_difficulty_num_nodes), dtype=np.int32),
        #     "spanning_tree": spaces.Box(low=0, high=1, shape=(self.max_difficulty_num_nodes, self.max_difficulty_num_nodes), dtype=np.int32),
        #     "weights": spaces.Box(low=0, high=10, shape=(self.max_difficulty_num_nodes, self.max_difficulty_num_nodes), dtype=np.float32),
        #     "attacked": spaces.MultiBinary(self.max_difficulty_num_nodes),
        #     # "action_mask": spaces.MultiBinary(self.max_difficulty_max_num_edges)
        # })

        # # Define the action space
        # self.action_space = spaces.MultiDiscrete([
        #     2,  # 0 = remove, 1 = add
        #     self.max_difficulty_num_nodes,  # index for node1
        #     self.max_difficulty_num_nodes   # index for node2
        # ])


        # # Create a MultiBinary space for each node's one-hot vector
        # one_hot_action_space = spaces.MultiBinary(self.max_difficulty_num_nodes)

        # # Combine these into a Tuple, one for each node in the pair
        # self.action_space = spaces.Tuple((one_hot_action_space, one_hot_action_space))

        # Adjusting the action space to be two discrete spaces using MultiDiscrete
        # self.action_space = spaces.MultiDiscrete([self.max_difficulty_num_nodes, self.max_difficulty_num_nodes])

        # Define the maximum number of edges in the graph (upper triangle only)
        max_edges = int(self.max_difficulty_num_nodes * (self.max_difficulty_num_nodes - 1) / 2)

        # Action space where each action is a binary decision (0 or 1) for every possible edge
        self.action_space = spaces.MultiBinary(max_edges)

        # # Create low and high arrays with the same shape as the node features
        # low = np.zeros((self.max_difficulty_num_nodes, 2), dtype=np.int32)  
        # high = np.zeros((self.max_difficulty_num_nodes, 2), dtype=np.int32)  ## Low values for both features High values for both features

        # # Set the range for each feature
        # # First feature (attacked status) ranges from 0 to 1
        # high[:, 0] = 1  # Maximum value for the attacked status
        # # Second feature (node index) ranges from 0 to max_difficulty_num_nodes - 1
        # high[:, 1] = self.max_difficulty_num_nodes - 1  # Maximum value for node index

        # # Define the space with correct low and high arrays
        # node_features_space = spaces.Box(low=low, high=high, dtype=np.int32)
        node_features_space = spaces.Box(low=0, high=1, shape=(self.max_difficulty_num_nodes, 5), dtype=np.int32)

        # Physical Edge Indices List
        physical_edge_indices_space = spaces.Box(low=0, high=self.max_difficulty_num_nodes-1, shape=(self.max_difficulty_max_num_edges, 2), dtype=np.int32)
        # Physical edge weights, assuming weight value max of 100
        physical_edge_weights_space = spaces.Box(low=0, high=100, shape=(self.max_difficulty_max_num_edges, 1), dtype=np.float32)

        # Spanning Tree Edge Indices List
        spanning_tree_edge_indices_space = spaces.Box(low=0, high=self.max_difficulty_num_nodes-1, shape=(self.max_difficulty_max_num_edges, 2), dtype=np.int32)
        # Spanning Tree Edge weights, assuming weight value max of 100
        spanning_tree_edge_weights_space = spaces.Box(low=0, high=100, shape=(self.max_difficulty_max_num_edges, 1), dtype=np.float32)
        
        # Edge masks to indicate real or padded edges
        physical_edge_mask_space = spaces.Box(low=0, high=1, shape=(self.max_difficulty_max_num_edges, 1), dtype=np.uint8)
        spanning_tree_edge_mask_space = spaces.Box(low=0, high=1, shape=(self.max_difficulty_max_num_edges, 1), dtype=np.uint8)

        # Mask to communicate what actions are allowed to be taken
        valid_action_mask = spaces.Box(low=0, high=1, shape=(self.max_difficulty_num_nodes, self.max_difficulty_num_nodes), dtype=np.uint8)
        # Define the first node choice mask to indicate valid first nodes
        first_node_action_mask = spaces.Box(low=0, high=1, shape=(self.max_difficulty_num_nodes,), dtype=np.uint8)

        self.observation_space = spaces.Dict({
            "physical_node_features": node_features_space,
            "spanning_node_features": node_features_space,
            "physical_edge_indices": physical_edge_indices_space,
            "physical_edge_weights": physical_edge_weights_space,
            "physical_edge_mask": physical_edge_mask_space,
            "spanning_tree_edge_indices": spanning_tree_edge_indices_space,
            "spanning_tree_edge_weights": spanning_tree_edge_weights_space,
            "spanning_tree_edge_mask": spanning_tree_edge_mask_space,
            "action_mask": valid_action_mask, 
            "first_node_action_mask": first_node_action_mask
        })

        # Initialize placeholder for node positions
        self.pos = None

        # Set of nodes that are attacked
        self.attacked_nodes = set()  

        # TODO: Temporary fix to make sure we generate a valid graph where every node is connected. 
        while True:
            # Create Physical network that will not change for the full duration of training!
            # TODO: Save this out so you can run inference on the same physical network
            # Create a new network environment for each episode
            self.network_env = NetworkEnvironment(self.min_nodes, self.max_nodes, self.min_redundancy)
            
            # Reset the network environment and get the initial network
            self.network = self.network_env.reset()

            # Retrieve positions after reset
            self.pos = self.network_env.get_positions() 

            # Get the number of nodes in the current network
            self.num_nodes = self.network_env.num_nodes
            # Keep track of number of nodes in each env
            self.num_nodes_history.append(self.num_nodes)

            # Create the action mask 
            self.action_mask = self.create_initial_action_mask(self.network, self.num_nodes)

            # Check if any node has no edges (i.e., if any row in the action mask is all zeros)
            if np.any(np.sum(self.action_mask, axis=1) == 0):
                # print("Detected a node with no edges, regenerating the network...")
                continue  # Regenerate the graph if a node without edges is detected

            # If the check passes, break the loop and proceed with the environment reset
            break

        if render_mode: 
            # Set up the Tkinter root window
            self.root = tk.Tk()
            self.root.wm_title("Spanning Tree Environment")
            
            # Set up Matplotlib figure and axes
            self.fig, self.ax = plt.subplots(1, 3, figsize=(14, 6))
            
            # Embed the Matplotlib figure in the Tkinter canvas
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def update_level_parameters(self):
        self.max_nodes = 4 + NUM_NODE_INCREASE_RATE_PER_LEVEL * self.current_level
        self.max_attacked_nodes = 1 + NUM_ATTACKED_NODE_INCREASE_RATE_PER_LEVEL * (self.current_level // 5)

    def should_level_up(self):

        # Don't go past a certain level otherwise the action space is too small
        if self.current_level >= self.final_difficulty_level:
            return False

        # Check performance history to decide on leveling up
        if self.current_level_total_timesteps >= self.num_timestep_cooldown:
            # Access as a property, not a method call
            average_performance = self.get_level_average_performance
            return average_performance > self.performance_threshold
        return False
    
    @property
    def get_level_average_performance(self):

        if len(self.reward_history) == 0: 
            return 0 
            
        return sum(self.reward_history) / len(self.reward_history)

    def reset(self, seed=None):

        # Store last episode's cumulative reward
        self.reward_history.append(self.ep_cumulative_reward)  
       
        # Reset cumulative reward at the start of an episode
        self.ep_cumulative_reward = 0

        # Set the random seed
        if seed is not None:
            np.random.seed(seed)

        # Evaluate if it's time to level up before resetting
        if self.should_level_up():
            self.current_level += 1
            # Update the env characteristics such as min and max nodes
            self.update_level_parameters()
            # Reset reward history
            self.reward_history = collections.deque(maxlen=self.history_size) 
            # Reset level timestep count
            self.current_level_total_timesteps = 0 
            print(f"Leveling up! Now at Level {self.current_level}")
            print(f"Max number of nodes at this level: {self.max_nodes}")
            print(f"Average number of nodes in the past level: {sum(self.num_nodes_history)/ len(self.num_nodes_history)}")
            self.num_nodes_history = [] # Reset num nodes history

        # Reset timestep
        self.current_step = 0 

        # # TODO: Temporary fix to make sure we generate a valid graph where every node is connected. 
        # while True: 
        #     # Create a new network environment for each episode
        #     self.network_env = NetworkEnvironment(self.min_nodes, self.max_nodes, self.min_redundancy)
            
        #     # Reset the network environment and get the initial network
        #     self.network = self.network_env.reset()

        #     # Retrieve positions after reset
        #     self.pos = self.network_env.get_positions() 

        #     # Clear the previous spanning tree if it exists
        #     if self.tree is not None:
        #         self.tree.clear()
            
        #     # # Compute the Minimum Spanning Tree of the network using the weights
        #     # self.tree = nx.minimum_spanning_tree(self.network, weight='weight')  

        #     # Initialize an empty tree with the same nodes as the network
        #     self.tree = nx.Graph()
        #     self.tree.add_nodes_from(self.network.nodes(data=True))
            
        #     # Get the number of nodes in the current network
        #     self.num_nodes = self.network_env.num_nodes
        #     # Keep track of number of nodes in each env
        #     self.num_nodes_history.append(self.num_nodes)

        #     # Create the action mask 
        #     self.action_mask = self.create_initial_action_mask(self.network, self.num_nodes)

        #     # Check if any node has no edges (i.e., if any row in the action mask is all zeros)
        #     if np.any(np.sum(self.action_mask, axis=1) == 0):
        #         # print("Detected a node with no edges, regenerating the network...")
        #         continue  # Regenerate the graph if a node without edges is detected

        #     # If the check passes, break the loop and proceed with the environment reset
        #     break


        # Clear the previous spanning tree if it exists
        if self.tree is not None:
            self.tree.clear()
        
        # Compute the Minimum Spanning Tree of the network using the weights
        self.tree = nx.minimum_spanning_tree(self.network, weight='weight')  


        # Simulate attack
        self.simulate_attack()

        # Calculate max number of edges for this network
        self.current_network_max_num_edges = int(self.num_nodes * (self.num_nodes - 1) / 2) 

        # Return the initial state
        return self.get_state(), {}

    def get_state(self):
        '''
        Attention! 
            If you ever try to get the adj. matrix:
            nx.to_numpy_array(self.network, weight=None, dtype=int) 
            WEIGHT MUST BE SET TO NONE!
        '''
        size = self.max_difficulty_num_nodes
        max_edges = self.max_difficulty_max_num_edges

        # Initialize node features arrays for physical network and spanning tree
        physical_node_features = np.zeros((size, 5), dtype=np.float32)
        spanning_node_features = np.zeros((size, 5), dtype=np.float32)
            
        # Set attacked status for node features
        for node in self.attacked_nodes:
            physical_node_features[node][0] = 1
            spanning_node_features[node][0] = 1

        # Compute node features for physical network
        degrees = np.array([d for _, d in self.network.degree()], dtype=np.float32)
        clustering = np.array([c for _, c in nx.clustering(self.network).items()], dtype=np.float32)
        eigenvector_centrality = np.array([e for _, e in nx.eigenvector_centrality_numpy(self.network).items()], dtype=np.float32)
        betweenness_centrality = np.array([b for _, b in nx.betweenness_centrality(self.network).items()], dtype=np.float32)

        for node in range(self.num_nodes):
            physical_node_features[node, 1] = degrees[node] if node < len(degrees) else 0
            physical_node_features[node, 2] = clustering[node] if node < len(clustering) else 0
            physical_node_features[node, 3] = eigenvector_centrality[node] if node < len(eigenvector_centrality) else 0
            physical_node_features[node, 4] = betweenness_centrality[node] if node < len(betweenness_centrality) else 0

        # Prepare padded arrays for edge indices, weights, and masks
        physical_network_edges_indices = np.zeros((max_edges, 2), dtype=np.int32)
        physical_network_weights = np.zeros((max_edges, 1), dtype=np.float32)
        physical_edge_mask = np.zeros((max_edges, 1), dtype=np.uint8)

        spanning_tree_edges_indices = np.zeros((max_edges, 2), dtype=np.int32)
        spanning_tree_weights = np.zeros((max_edges, 1), dtype=np.float32)
        spanning_tree_edge_mask = np.zeros((max_edges, 1), dtype=np.uint8)

        # Fill actual data for physical network 
        actual_physical_edges = np.array([[u, v] for u, v in self.network.edges()], dtype=np.int32)
        actual_physical_weights = np.array([[self.network.edges[u, v]['weight']] for u, v in self.network.edges()], dtype=np.float32)
        physical_network_edges_indices[:actual_physical_edges.shape[0]] = actual_physical_edges
        physical_network_weights[:actual_physical_edges.shape[0]] = actual_physical_weights

        # Add a mask for the physical network
        physical_edge_mask[:actual_physical_edges.shape[0], 0] = 1  

        # Fill actual data for the spanning tree, if available
        if self.tree.number_of_edges() > 0:
            # Compute node features for physical network
            degrees = np.array([d for _, d in self.tree.degree()], dtype=np.float32)
            clustering = np.array([c for _, c in nx.clustering(self.tree).items()], dtype=np.float32)
            eigenvector_centrality = np.array([e for _, e in nx.eigenvector_centrality_numpy(self.network).items()], dtype=np.float32)
            betweenness_centrality = np.array([b for _, b in nx.betweenness_centrality(self.tree).items()], dtype=np.float32)

            for node in range(self.num_nodes):
                spanning_node_features[node, 1] = degrees[node] if node < len(degrees) else 0
                spanning_node_features[node, 2] = clustering[node] if node < len(clustering) else 0
                spanning_node_features[node, 3] = eigenvector_centrality[node] if node < len(eigenvector_centrality) else 0
                spanning_node_features[node, 4] = betweenness_centrality[node] if node < len(betweenness_centrality) else 0

            # Fill actual data for the spanning tree
            actual_spanning_tree_edges = np.array([[u, v] for u, v in self.tree.edges()], dtype=np.int32)
            actual_spanning_tree_weights = np.array([[self.tree.edges[u, v]['weight']] for u, v in self.tree.edges()], dtype=np.float32)
            spanning_tree_edges_indices[:actual_spanning_tree_edges.shape[0]] = actual_spanning_tree_edges
            spanning_tree_weights[:actual_spanning_tree_edges.shape[0]] = actual_spanning_tree_weights

            # Add mask for the spanning tree
            spanning_tree_edge_mask[:actual_spanning_tree_edges.shape[0], 0] = 1 

        # Define first node choice mask (1 for nodes in the spanning tree, 0 otherwise)
        self.first_node_action_mask = np.zeros(size, dtype=np.uint8)
        for node in self.tree.nodes():
            if self.tree.degree[node] > 0:  # Check if the node has any edges in the spanning tree
                self.first_node_action_mask[node] = 1

        return {
            "physical_node_features": physical_node_features,
            "spanning_node_features": spanning_node_features,
            "physical_edge_indices": physical_network_edges_indices,
            "physical_edge_weights": physical_network_weights,
            "physical_edge_mask": physical_edge_mask,
            "spanning_tree_edge_indices": spanning_tree_edges_indices,
            "spanning_tree_edge_weights": spanning_tree_weights,
            "spanning_tree_edge_mask": spanning_tree_edge_mask,
            "action_mask": self.action_mask, 
            "first_node_action_mask": self.first_node_action_mask,
        }

    def create_initial_action_mask(self, network, num_nodes):
        mask = np.zeros((num_nodes, num_nodes), dtype=int)
        for i in range(num_nodes):
             # Only upper triangle needed for undirected graph
            for j in range(i + 1, num_nodes): 
                if network.has_edge(i, j):
                    # Upper triangle 
                    mask[i][j] = 1
                    # Since undirected graph for now, keep matrix symmetrical. 
                    mask[j][i] = 1

        return mask

    def update_action_mask(self, network, mask, node1, node2, action):
        '''
            If a connection was added between two nodes then remove
            the '1' in that place indicating that it is no longer a 
            valid action to take. If connection is removed between 
            two nodes then add back the '1' to indicate that it is a 
            valid action now.
        '''
        if action == 'add':
            mask[node1][node2] = 0
            mask[node2][node1] = 0
        # TODO: Make sure this maintains upper traingle
        elif action == 'remove' and network.has_edge(node1, node2):
            mask[node1][node2] = 1
            mask[node2][node1] = 1

    def process_actions(self, continuous_actions, threshold=0.5):
        # Convert continuous actions into binary decisions
        binary_actions = (continuous_actions > threshold).astype(int)
        return binary_actions

    def step(self, action):   

        # Convert continuous actions to binary decisions
        # action = self.process_actions(action) 

        # Execute the action
        valid_action, invalid_action, connected_to_attacked_node, disconnected_from_attacked_node = self.execute_action(action)
        
        # Calculate reward and check if the goal is achieved (done)
        reward, done = self.calculate_reward(valid_action, invalid_action, connected_to_attacked_node, disconnected_from_attacked_node)

        # Initialize truncated as False
        truncated = False
        
        # Check if the maximum steps have been reached
        if self.current_step >= self.max_ep_steps:
            truncated = True  # Truncate the episode due to step count limit
            done = True  # Still set done to True to end the episode

        # Increment timestep
        self.current_step += 1
        # Increment level timestep
        self.current_level_total_timesteps +=1

        # Render the current state of the environment if required
        if self.render_mode:
            self.render()
            self.root.update()

        # Keep track of the rewards for this episode
        self.ep_cumulative_reward += reward

        return self.get_state(), reward, done, truncated, {}

    def execute_action(self, action):
        valid_action = 0
        invalid_action = 0
        connected_to_attacked_node = 0
        disconnected_from_attacked_node = 0  # Optional for removals

        # Iterate over all possible edges by looping through node pairs
        action_index = 0  # To keep track of the current position in the action list
        for node1 in range(self.num_nodes):
            for node2 in range(node1 + 1, self.num_nodes):  # Ensure we only check each edge once (node1 < node2)
                
                # If the action for this edge is 1 (i.e., the agent wants to add the edge)
                if action[action_index] == 1:
                    if not self.tree.has_edge(node1, node2):
                        if self.network.has_edge(node1, node2):
                            # Add the edge to the spanning tree
                            self.tree.add_edge(node1, node2, weight=self.network[node1][node2]['weight'])
                            self.update_action_mask(self.network, self.action_mask, node1, node2, 'add')

                            # # Check if adding this edge creates a valid tree (i.e., no cycles)
                            # subgraph_with_edges = self.tree.subgraph([n for n in self.tree.nodes if self.tree.degree(n) > 0])
                            # if nx.is_tree(subgraph_with_edges):
                            #     valid_action += 1  # Mark this as a valid action
                            #     self.update_action_mask(self.network, self.action_mask, node1, node2, 'add')
                            # else:
                            #     # If adding the edge creates a cycle, remove it
                            #     self.tree.remove_edge(node1, node2)
                            #     invalid_action += 1  # Mark this as an invalid action

                # Optionally, handle removing edges if allowed
                elif action[action_index] == 0 and self.tree.has_edge(node1, node2):
                    self.tree.remove_edge(node1, node2)
                    self.update_action_mask(self.network, self.action_mask, node1, node2, 'remove')

                # Increment the action index as we move through the list
                action_index += 1

        return valid_action, invalid_action, connected_to_attacked_node, disconnected_from_attacked_node


    # def execute_action(self, action):
    #     # Decode the one-hot encoded node vectors
    #     node1, node2 = action

    #     valid_action = 0
    #     invalid_action = 0
    #     connected_to_attacked_node = 0
    #     disconnected_from_attacked_node = 0  # This may not be needed if removing is not allowed

    #     # Ensure the nodes are within the current graph's bounds and node1 is not the same as node2
    #     if node1 < self.num_nodes and node2 < self.num_nodes and node1 != node2:
    #         # Only allow adding edges since removing isn't an option
    #         if not self.tree.has_edge(node1, node2):
    #             if self.network.has_edge(node1, node2):
    #                 # Temporarily add the edge to check if it would create a valid tree
    #                 self.tree.add_edge(node1, node2, weight=self.network[node1][node2]['weight'])
                
    #                 # Get the subgraph that only includes nodes with at least one edge
    #                 nodes_with_edges = [n for n in self.tree.nodes if self.tree.degree(n) > 0]
    #                 subgraph_with_edges = self.tree.subgraph(nodes_with_edges)

    #                 # Check if the resulting graph is still a tree
    #                 if nx.is_tree(subgraph_with_edges):
    #                     valid_action = 1  # Mark this as a valid action
    #                     # Keep track of addition in mask
    #                     self.update_action_mask(self.network, self.action_mask, node1, node2, 'add')
    #                 else:
    #                     # If adding the edge would create a cycle, remove it
    #                     self.tree.remove_edge(node1, node2)
    #                     invalid_action = 1  # Mark this as an invalid action

    #     return valid_action, invalid_action, connected_to_attacked_node, disconnected_from_attacked_node

    def decode_action(self, action):
        node1 = np.argmax(action[0])  # Assuming action[0] is the one-hot vector for node1
        node2 = np.argmax(action[1])  # Assuming action[1] is the one-hot vector for node2
        print(f"NODE 1: {node1} NODE 2: {node2}")
        return node1, node2

    def is_connection_to_attacked_node(self, node1, node2):
        return node1 in self.attacked_nodes or node2 in self.attacked_nodes

    def calculate_reward(self, valid_action, invalid_action, connected_to_attacked_node, disconnected_from_attacked_node):
        
        reward = -10  
        done = False

        # Get the subgraph excluding attacked nodes
        non_attacked_subgraph = self.tree.subgraph([n for n in self.tree.nodes if n not in self.attacked_nodes])

        # Penalty for each step (negative reward)
        reward -= 0.1  

        # Check if the non-attacked subgraph is a tree and connected
        is_tree = nx.is_tree(non_attacked_subgraph)

        # Reward reduction for nodes that are part of the tree (i.e., they have exactly 1 parent)
        nodes_with_one_parent = len([n for n in non_attacked_subgraph.nodes if non_attacked_subgraph.degree(n) == 1])

        # Gradually reduce the negative reward as more nodes form proper tree structures
        reward += nodes_with_one_parent * 0.5

        # Slightly reduce the negative reward if the attacked nodes are isolated
        all_isolated = self.is_attacked_isolated()
        if all_isolated:
            reward += 1  # Reduce negative reward for isolation of attacked nodes

        # Large positive reward if the non-attacked nodes form a connected tree and all attacked nodes are isolated
        if all_isolated and is_tree:
            reward = 10  # Large positive reward for completing the task
            done = True

        # if invalid_action:
        #     reward = -1

        # # Get the subgraph that only includes nodes with at least one edge
        # nodes_with_edges = [n for n in self.tree.nodes if self.tree.degree(n) > 0]
        # subgraph_with_edges = self.tree.subgraph(nodes_with_edges)

        # # Check if the subgraph with edges forms a tree
        # if not nx.is_tree(subgraph_with_edges):
        #     # If a loop is detected, terminate the episode
        #     done = True
        #     reward = -100
        #     return reward, done
    
        # # Check if any attacked nodes are in the tree
        # attacked_nodes_in_tree = [n for n in self.attacked_nodes if n in subgraph_with_edges.nodes]
        # if attacked_nodes_in_tree:
        #     reward -= 300
        #     done = True
        #     return reward, done
           
        # # Penalty for connecting to attacked nodes
        # reward -= .1 * connected_to_attacked_node  

        # # Reduced reward for disconnecting attacked nodes
        # reward += 0.05 * disconnected_from_attacked_node  

        # all_isolated = self.is_attacked_isolated()

        # if all_isolated:
        #     reward = -.01
        #     non_attacked_subgraph = self.tree.subgraph([n for n in self.tree.nodes if n not in self.attacked_nodes])
        #     if nx.is_tree(non_attacked_subgraph) and nx.is_connected(non_attacked_subgraph):
        #         # tree_weight = sum(data['weight'] for u, v, data in non_attacked_subgraph.edges(data=True))
        #         # # Major reward for completing the main objective
        #         # reward += 50 - 0.1 * tree_weight  
        #         # # Stronger bonus for early completion
        #         # reward += 0.5 * (self.max_ep_steps - self.current_step)  
        #         reward = 5
        #         done = True
        #         return reward, done

        # # # Terminate the episode if no more valid actions are possible
        # # if not self.action_mask.any():
        # #     done = True  
            
        return reward, done


    def is_attacked_isolated(self):
        # Check each attacked node to see if it is completely isolated
        for node in self.attacked_nodes:
            if any(self.tree.has_edge(node, other) for other in self.tree.nodes if other != node):
                return False
        return True

    def render(self, mode='human'):
        # Clear the previous plots
        self.ax[0].clear()
        self.ax[1].clear()
        self.ax[2].clear()
        
        # Draw the original physical network
        nx.draw(self.network, self.pos, with_labels=True, node_color='skyblue', node_size=self.node_size, edge_color='gray', ax=self.ax[0])
        self.ax[0].set_title("Original Physical Network")
        
        # Draw the current spanning tree
        nx.draw(self.tree, self.pos, with_labels=True, node_color='lightgreen', node_size=self.node_size, edge_color='gray', ax=self.ax[1]) 
        self.ax[1].set_title("Spanning Tree")

        # Attacked Spanning Tree
        node_colors = ['red' if node in self.attacked_nodes else 'lightgreen' for node in self.tree.nodes()]
        nx.draw(self.tree, self.pos, with_labels=True, node_color=node_colors, node_size=self.node_size, edge_color='gray', ax=self.ax[2])
        self.ax[2].set_title("Attacked Spanning Tree")

        # Check if weight labels should be shown
        if self.show_weight_labels:
            # Label for original physical network
            edge_labels = nx.get_edge_attributes(self.network, 'weight')
            nx.draw_networkx_edge_labels(self.network, self.pos, edge_labels=edge_labels, ax=self.ax[0])

            # Label for current spanning tree
            edge_labels = nx.get_edge_attributes(self.tree, 'weight')
            nx.draw_networkx_edge_labels(self.tree, self.pos, edge_labels=edge_labels, ax=self.ax[1])

        # Update the canvas to reflect the new plots
        self.canvas.draw()

    def close(self):
        # Properly close the Tkinter window
        self.root.quit()
        self.root.destroy()

    def simulate_attack(self):
        # TODO: Move to separate class
        # TODO: Vary the number of attacked nodes
        # TODO: Intelligent choice of nodes to attack

        # Find max possible attacks since num nodes might be less than number of attacks
        possible_attacks = min(self.num_nodes - 2, self.max_attacked_nodes)

        # Randomly decide the number of nodes to attack within the given range
        self.num_attacked_nodes = np.random.randint(self.min_attacked_nodes, possible_attacks + 1)

        # Randomly select a few nodes to attack
        self.attacked_nodes = set(np.random.choice(self.network.nodes(), self.num_attacked_nodes, replace=False))
        
# Example usage
if __name__ == "__main__":
    # Create the SpanningTreeEnv environment
    env = SpanningTreeEnv(min_nodes=5, 
                          max_nodes=5, 
                          min_redundancy=3, 
                          min_attacked_nodes=1, 
                          max_attacked_nodes=2,
                          start_difficulty_level=1,
                          final_difficulty_level=1,
                          num_timestep_cooldown=2, 
                          show_weight_labels=SHOW_WEIGHT_LABELS, 
                          render_mode=True, 
                          max_ep_steps=100, 
                          node_size=250, 
                          performance_threshold=-1)
    
    # Reset the environment to start a new episode
    state = env.reset()
    done = False
    
    # Run the simulation loop until the episode is done
    while True:
        # Select a random action from the action space
        action = env.action_space.sample()

        # Execute the action and get the new state, reward, and done flag
        state, reward, done, _, _ = env.step(action)
        print(f' Action: \n {action} , Reward {reward}')

        time.sleep(.1)
        
        # Update the Tkinter window
        env.root.update()

        if done:
            state = env.reset()
            done = False
    
    print("Done")
    time.sleep(30)

    # Close the environment
    env.close()



# # Function to pad matrices to the maximum node size
# size = self.max_difficulty_num_nodes

# # Convert the full network and MST to adjacency matrices
# full_net_matrix = nx.to_numpy_array(self.network, weight=None, dtype=int) # WEIGHT MUST BE SET TO NONE!
# mst_matrix = nx.to_numpy_array(self.tree, weight=None, dtype=int)         # WEIGHT MUST BE SET TO NONE!

# # Pad the full network matrix
# full_net_matrix_padded = np.zeros((size, size), dtype=int)
# full_net_matrix_padded[:full_net_matrix.shape[0], :full_net_matrix.shape[1]] = full_net_matrix

# # Pad the MST matrix
# mst_matrix_padded = np.zeros((size, size), dtype=int)
# mst_matrix_padded[:mst_matrix.shape[0], :mst_matrix.shape[1]] = mst_matrix

# # Extract edge weights from the full network and pad it
# weights_matrix = nx.to_numpy_array(self.network, weight='weight')
# weights_matrix_padded = np.zeros((size, size), dtype=float)
# weights_matrix_padded[:weights_matrix.shape[0], :weights_matrix.shape[1]] = weights_matrix

# # Create an adjacency matrix where attacked nodes are marked
# attacked_vector = np.zeros(size, dtype=int)
# for node in self.attacked_nodes:
#     attacked_vector[node] = 1

# # Get action mask
# action_mask = self.get_action_mask()

# # Complete State: Ensure all parts are padded to the maximum size
# return {
#     "full_network": full_net_matrix_padded,
#     "mst": mst_matrix_padded,
#     "weights": weights_matrix_padded,
#     "attacked": attacked_vector,
#     "action_mask": action_mask,
# }

#  def execute_action(self, action):

#         # Could be adding edge or removing edge
#         valid_action = 0
#         # Adding edge when physical network doesn't have edge
#         invalid_action = 0
#         # Added a connection to attacked node
#         connected_to_attacked_node = 0 
#         # Removed connection to attacked node
#         disconnected_from_attacked_node = 0

#         # Iterate over all possible connections within the current network size 
#         index = 0
#         # Iterate over only the number of nodes in this current network size
#         for i in range(self.num_nodes):
#             # lower triangle not needed in full matrix approach
#             for j in range(i + 1, self.num_nodes):
#                 # Check the action for the edge (i, j)
#                 if action[index] == 1 and not self.tree.has_edge(i, j):
#                     if self.network.has_edge(i, j):
#                         self.tree.add_edge(i, j, weight=self.network[i][j]['weight'])
#                         valid_action += 1
#                     else:
#                         invalid_action += 1

#                     if i in self.attacked_nodes or j in self.attacked_nodes:
#                         connected_to_attacked_node += 1

#                 elif action[index] == 0 and self.tree.has_edge(i, j):
#                     self.tree.remove_edge(i, j)
#                     if i in self.attacked_nodes or j in self.attacked_nodes:
#                         disconnected_from_attacked_node += 1
#                 index += 1

#         return valid_action, invalid_action, connected_to_attacked_node, disconnected_from_attacked_node


# def get_action_mask(self):
    
#     # Initialize mask with zeros
#     action_mask = np.zeros(self.action_space.shape[0], dtype=int)
#     # Activate the mask for the valid actions based on 
#     # number of valid edges (actions) for the current number of active nodes
#     action_mask[:self.current_network_max_num_edges] = 1
    
#     return action_mask


# def execute_action(self, action):
#     # Unpack the action tuple
#     action_type, node1, node2 = action[0], action[1], action[2] 

#     valid_action = 0
#     invalid_action = 0
#     connected_to_attacked_node = 0
#     disconnected_from_attacked_node = 0

#     # Ensure the nodes are within the current graph's bounds and node1 is not the same as node2
#     if node1 < self.num_nodes and node2 < self.num_nodes and node1 != node2:
#         # Action to add an edge
#         if action_type == 1:  
#             # Check if the edge does not already exist
#             if not self.tree.has_edge(node1, node2):  
#                 # Check if it's a valid edge in the physical network
#                 if self.network.has_edge(node1, node2):  
#                     # Add edge in the "spanning tree"
#                     self.tree.add_edge(node1, node2, weight=self.network[node1][node2]['weight'])
#                     valid_action += 1
#                     # Check if either of the nodes are attacked nodes
#                     if self.is_connection_to_attacked_node(node1, node2):
#                         connected_to_attacked_node += 1
#                 else:
#                     # Trying to add a non-existent edge in the physical network
#                     invalid_action += 1 
#         # Action to remove an edge 
#         elif action_type == 0:  
#             # Check if the edge exists
#             if self.tree.has_edge(node1, node2):  
#                 self.tree.remove_edge(node1, node2)
#                 valid_action += 1
#                 # Check if either of the nodes are attacked nodes
#                 if self.is_connection_to_attacked_node(node1, node2):
#                     disconnected_from_attacked_node += 1
#             else:
#                 invalid_action += 1  # Trying to remove a non-existent edge in the tree
#     else:
#         invalid_action += 1  # Invalid node indices

#     return valid_action, invalid_action, connected_to_attacked_node, disconnected_from_attacked_node