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
from safe_ptp.src.ptp.clock_sim import ClockSimulation
from safe_ptp.src.env.node_attacker import NodeAttacker

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
    
        # Set of nodes that are attacked
        # self.attacked_nodes = set() 
        self.node_attacker = NodeAttacker()
        self.malicious_nodes = [] 

        # Create a PTP Simulation instance
        self.clock_sim = ClockSimulation(community_size=7, community_num=7, render=self.render_mode, seed=40)

        # Get the number of undirected edges in the physical network graph
        self.num_physical_edges = self.clock_sim.graph.number_of_edges()

        # Get the number of nodes in the physical network graph
        self.num_physical_nodes = self.clock_sim.graph.number_of_nodes()

        # Action space where each action is a binary decision (0 or 1) for every possible edge
        self.action_space = spaces.MultiBinary(2*self.num_physical_edges) # Double num edges because directed graph

        # # Define the space with correct low and high arrays
        # node_features_space = spaces.Box(low=low, high=high, dtype=np.int32)
        node_features_space = spaces.Box(low=0, high=100000, shape=(self.num_physical_nodes, 10), dtype=np.int32)

        # Spanning Tree Edge Indices List
        spanning_tree_edge_indices_space = spaces.Box(low=0, high=self.num_physical_nodes-1, shape=(2*self.num_physical_edges, 2), dtype=np.int32)

        # Edge masks to indicate real or padded edges
        spanning_tree_edge_mask_space = spaces.Box(low=0, high=1, shape=(2*self.num_physical_edges, 1), dtype=np.uint8)

        # Previous tree
        previous_action = spaces.MultiBinary(2*self.num_physical_edges) # Double num edges because directed graph

        self.observation_space = spaces.Dict({
            "node_features": node_features_space,
            "spanning_tree_edge_indices": spanning_tree_edge_indices_space,
            "spanning_tree_edge_mask": spanning_tree_edge_mask_space,
            "previous_action": previous_action
        })

        # if render_mode: 
        #     # Set up the Tkinter root window
        #     self.root = tk.Tk()
        #     self.root.wm_title("Spanning Tree Environment")
            
        #     # Set up Matplotlib figure and axes
        #     self.fig, self.ax = plt.subplots(1, 3, figsize=(14, 6))
            
        #     # Embed the Matplotlib figure in the Tkinter canvas
        #     self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        #     self.canvas.draw()
        #     self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

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

        self.clock_sim.reset()

        # Select and set malicious nodes using the attacker class
        self.select_malicious_nodes(self.clock_sim.tree, self.clock_sim.leader_node, num_malicious=2)
        
        # Set the attacked nodes attributes    
        self.clock_sim.set_malicious_attributes(self.malicious_nodes)

        # Simulate network with stating configuration
        self.clock_sim.simulate_and_render(sync_interval=5, steps=20)

        # Return the initial state
        return self.get_state(), {}

    def get_state(self):
        '''
        Attention! 
            If you ever try to get the adj. matrix:
            nx.to_numpy_array(self.network, weight=None, dtype=int) 
            WEIGHT MUST BE SET TO NONE!
        '''

        # Initialize node features arrays for physical network and spanning tree
        node_features = np.zeros((self.num_physical_nodes, 10), dtype=np.float32)
        
        # Get the features from the clock simulation environment
        state_features = self.clock_sim.get_state_features()

        scaling_factor = 0.001

        # Now, we need to map the state_features into node_features for our environment
        for i in range(self.num_physical_nodes):
            # Fill node features from the state features:
            node_features[i, 0] = state_features['has_malicious_ancestor'][i]
            node_features[i, 1] = state_features['is_malicious'][i]
            node_features[i, 2] = state_features['disconnected'][i]
            node_features[i, 3] = state_features['susceptibility'][i]*scaling_factor
            node_features[i, 4] = state_features['time'][i]*scaling_factor
            node_features[i, 5] = state_features['drift'][i]*scaling_factor
            node_features[i, 6] = state_features['hops'][i]

            # For type, we split up the one-hot encoding
            node_features[i, 7:10] = state_features['type'][i]  # 3 elements for one-hot encoding

        # Prepare padded arrays for edge indices and masks
        spanning_tree_edges = np.zeros((self.num_physical_edges * 2, 2), dtype=np.int32)
        spanning_tree_edge_mask = np.zeros((self.num_physical_edges * 2, 1), dtype=np.uint8)

        # Fill actual data for the spanning tree
        actual_spanning_tree_edges = self.clock_sim.get_tree_edge_indices()
        spanning_tree_edges[:actual_spanning_tree_edges.shape[0]] = actual_spanning_tree_edges

        # Add mask for the spanning tree
        spanning_tree_edge_mask[:actual_spanning_tree_edges.shape[0], 0] = 1 

        # Get the validated edge vector from the previous step. This represents the tree in vector form
        previous_action = self.clock_sim.get_tree_as_edge_vector()
        
        # Create the observation space dict with node features, spanning tree edges, and mask
        return {
            "node_features": node_features,
            "spanning_tree_edge_indices": spanning_tree_edges,
            "spanning_tree_edge_mask": spanning_tree_edge_mask,
            "previous_action": previous_action, 
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

        # Reconfigure the tree using the action vector
        self.clock_sim.construct_tree_from_edge_vector(action)

        # Simulate PTP synchronization and visualize if rendering is enabled
        self.clock_sim.simulate_and_render(sync_interval=5, steps=20)

        # Calculate reward and check if the goal is achieved (done)
        reward, done = self.calculate_reward()

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

        # # Render the current state of the environment if required
        # if self.render_mode:
        #     self.render()
        #     self.root.update()

        return self.get_state(), reward, done, truncated, {}


    # def decode_action(self, action):
    #     node1 = np.argmax(action[0])  # Assuming action[0] is the one-hot vector for node1
    #     node2 = np.argmax(action[1])  # Assuming action[1] is the one-hot vector for node2
    #     print(f"NODE 1: {node1} NODE 2: {node2}")
    #     return node1, node2

    def is_connection_to_attacked_node(self, node1, node2):
        return node1 in self.attacked_nodes or node2 in self.attacked_nodes

    def calculate_reward(self):

        done = False

        # Keep track of the rewards for this episode
        self.ep_cumulative_reward +=  -self.clock_sim.get_total_desync_time() * 0.0001

        return self.ep_cumulative_reward, done


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

    def select_malicious_nodes(self, tree, leader_node, num_malicious):
        # TODO: Move to separate class
        # TODO: Vary the number of attacked nodes
        # TODO: Intelligent choice of nodes to attack

        self.malicious_nodes = self.node_attacker.select_malicious_nodes(tree, leader_node, num_malicious)


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

        print(state)
        time.sleep(.1)
        
        # # Update the Tkinter window
        # env.root.update()

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