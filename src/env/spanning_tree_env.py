import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter as tk
import time

from network_env import NetworkEnvironment

SHOW_WEIGHT_LABELS = False 

class SpanningTreeEnv(gym.Env):
    def __init__(self, min_nodes, 
                       max_nodes, 
                       min_redundancy, 
                       max_redundancy, 
                       show_weight_labels=False, 
                       render_mode=False, 
                       max_ep_steps=100, 
                       node_size=700):
        super(SpanningTreeEnv, self).__init__()
        
        # Initialize parameters for the network environment
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.min_redundancy = min_redundancy
        self.max_redundancy = max_redundancy

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
        
        # Initialize placeholders for the number of nodes, action space, and observation space
        self.num_nodes = max_nodes 

        # Define the action space as pairs of nodes (action_type ,parent, child)
        self.action_space = spaces.MultiDiscrete([2, self.num_nodes, self.num_nodes])

        # Define the observation space 
        # TODO currently observation space is max number of nodes. Explore embedding to an equal dimension.
        self.observation_space = spaces.Dict({
            "full_network": spaces.Box(low=0, high=1, shape=(self.num_nodes, self.num_nodes), dtype=np.int32),
            "mst": spaces.Box(low=0, high=1, shape=(self.num_nodes, self.num_nodes), dtype=np.int32),
            "weights": spaces.Box(low=0, high=1, shape=(self.num_nodes, self.num_nodes), dtype=np.int32),
            "attacked": spaces.Box(low=0, high=1, shape=(self.num_nodes,), dtype=np.int32)
        })

        # Initialize placeholder for node positions
        self.pos = None

        # Set of nodes that are attacked
        self.attacked_nodes = set()  

        # Set up the Tkinter root window
        self.root = tk.Tk()
        self.root.wm_title("Spanning Tree Environment")
        
        # Set up Matplotlib figure and axes
        self.fig, self.ax = plt.subplots(1, 3, figsize=(14, 6))
        
        # Embed the Matplotlib figure in the Tkinter canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def reset(self, seed=None):

        # Set the random seed
        if seed is not None:
            np.random.seed(seed)

        # Reset timestep
        self.current_step = 0 

        # Create a new network environment for each episode
        self.network_env = NetworkEnvironment(self.min_nodes, self.max_nodes, self.min_redundancy, self.max_redundancy)
        
        # Reset the network environment and get the initial network
        self.network = self.network_env.reset()

         # Retrieve positions after reset
        self.pos = self.network_env.get_positions(self.network) 

        # Clear the previous spanning tree if it exists
        if self.tree is not None:
            self.tree.clear()
        
        # Compute the Minimum Spanning Tree of the network using the weights
        self.tree = nx.minimum_spanning_tree(self.network, weight='weight')  
        
        # Get the number of nodes in the current network
        self.num_nodes = self.network_env.num_nodes

        # Simulate attack
        self.simulate_attack()

        # Return the initial state
        return self.get_state(), {}

    def get_state(self):

        # Function to pad matrices to the maximum node size
        size = self.max_nodes

        # Convert the full network and MST to adjacency matrices
        full_net_matrix = nx.to_numpy_array(self.network, dtype=int)
        mst_matrix = nx.to_numpy_array(self.tree, dtype=int)

        # Pad the full network matrix
        full_net_matrix_padded = np.zeros((size, size), dtype=int)
        full_net_matrix_padded[:full_net_matrix.shape[0], :full_net_matrix.shape[1]] = full_net_matrix

        # Pad the MST matrix
        mst_matrix_padded = np.zeros((size, size), dtype=int)
        mst_matrix_padded[:mst_matrix.shape[0], :mst_matrix.shape[1]] = mst_matrix

        # Extract edge weights from the full network and pad it
        weights_matrix = nx.to_numpy_array(self.network, weight='weight')
        weights_matrix_padded = np.zeros((size, size), dtype=int)
        weights_matrix_padded[:weights_matrix.shape[0], :weights_matrix.shape[1]] = weights_matrix

        # Create and pad the binary array indicating attacked nodes
        attacked_vector = np.zeros(size, dtype=int)
        for node in self.attacked_nodes:
            if node < size:
                attacked_vector[node] = 1

        # Complete State: Ensure all parts are padded to the maximum size
        return {
            "full_network": full_net_matrix_padded,
            "mst": mst_matrix_padded,
            "weights": weights_matrix_padded,
            "attacked": attacked_vector
        }
    
    def step(self, action):

        # Unpack the action tuple
        action_type, parent, child = action  

        # Default penalty for invalid actions
        reward = -1 
        done = False
        truncated = False

        # Attempt to add a connection
        if action_type == 0:  
            if parent in self.tree.nodes and child in self.tree.nodes and not self.tree.has_edge(parent, child) and self.network.has_edge(parent, child):
                self.tree.add_edge(parent, child, weight=self.network[parent][child]['weight'])
                # Reward for a valid action
                reward = .1 

         # Attempt to remove a connection
        elif action_type == 1: 
            if self.tree.has_edge(parent, child):
                self.tree.remove_edge(parent, child)
                # Reward for a valid action
                reward = .1  
                # Additional reward for removing a connection from an attacked node
                if parent in self.attacked_nodes or child in self.attacked_nodes:
                    reward += 1  

        # Check each attacked node if it is isolated
        for node in self.attacked_nodes:
            if all(not self.tree.has_edge(node, other) for other in self.tree.nodes if other != node):
                # Reward for completely isolating an attacked node
                reward += 1

        # Check if attacked nodes are isolated
        if all(not self.tree.has_edge(node, other) for node in self.attacked_nodes for other in self.tree.nodes if other != node):

            # Reward for isolating attacked nodes
            reward += 1

            # All attacked nodes are isolated, now check the remaining graph
            non_attacked_subgraph = self.tree.subgraph([n for n in self.tree.nodes if n not in self.attacked_nodes])

            # Check if subgraph is connected
            if nx.is_connected(non_attacked_subgraph):

                # Calculate the current total weight of the tree
                current_weight = sum(data['weight'] for u, v, data in non_attacked_subgraph.edges(data=True))
                # Encourage lighter trees
                # TODO. Normalize this since larger networks will have more cost
                reward += 10 - current_weight

                # Check if subgraph is a valid tree  
                if nx.is_tree(non_attacked_subgraph):
                    # Bonus for a valid spanning tree
                    reward += 10  
                    # End the episode if the tree is valid and connected
                    done = True  
                else:
                    # Penalize if not a valid tree
                    reward -= .2  
            else:
                # Penalize disconnection among non-attacked nodes
                reward -= .2  
        else:
            # Penalty for not isolating attacked nodes
            reward -= .1

        if self.current_step >= self.max_ep_steps:
            done = True  # End the episode because the max step count has been reached
            truncated = True

        # Increment timestep
        self.current_step += 1

        # Render the current state of the environment
        if self.render_mode:
            self.render()
            self.root.update()

        return self.get_state(), reward, done, truncated, {}


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

    def simulate_attack(self, num_attacks=2):
        # TODO: Move to separate class
        # TODO: Vary the number of attacked nodes
        # TODO: Intelligent choice of nodes to attack
        # Randomly select a few nodes to attack
        self.attacked_nodes = set(np.random.choice(self.network.nodes(), num_attacks, replace=False))
        
# Example usage
if __name__ == "__main__":
    # Create the SpanningTreeEnv environment
    env = SpanningTreeEnv(min_nodes=50, 
                          max_nodes=50, 
                          min_redundancy=2, 
                          max_redundancy=4, 
                          show_weight_labels=SHOW_WEIGHT_LABELS, 
                          render_mode=True, 
                          node_size=250)
    
    # Reset the environment to start a new episode
    state = env.reset()
    done = False
    
    # Run the simulation loop until the episode is done
    while not done:
        # Select a random action from the action space
        action = env.action_space.sample()

        # Execute the action and get the new state, reward, and done flag
        state, reward, done, _ = env.step(action)
        print(action)
        
        # Update the Tkinter window
        env.root.update()
    
    print("Done")
    time.sleep(30)

    # Close the environment
    env.close()