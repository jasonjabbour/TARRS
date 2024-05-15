import gym
from gym import spaces
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import tkinter as tk
import time

from network_env import NetworkEnvironment

class SpanningTreeEnv(gym.Env):
    def __init__(self, min_nodes, max_nodes, min_redundancy, max_redundancy, show_weight_labels=False):
        super(SpanningTreeEnv, self).__init__()
        
        # Initialize parameters for the network environment
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.min_redundancy = min_redundancy
        self.max_redundancy = max_redundancy

        # Parameter to control weight label rendering
        self.show_weight_labels = show_weight_labels 

        # Initialize placeholders for the network environment and graphs
        self.network_env = None
        self.network = None
        self.tree = None
        
        # Initialize placeholders for the number of nodes, action space, and observation space
        self.num_nodes = None
        self.action_space = None
        self.observation_space = None

        # Initialize placeholder for node positions
        self.pos = None
        
        # Set up the Tkinter root window
        self.root = tk.Tk()
        self.root.wm_title("Spanning Tree Environment")
        
        # Set up Matplotlib figure and axes
        self.fig, self.ax = plt.subplots(1, 2, figsize=(14, 6))
        
        # Embed the Matplotlib figure in the Tkinter canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def reset(self):        
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
        
        # Define the action space as pairs of nodes (parent, child)
        self.action_space = spaces.MultiDiscrete([self.num_nodes, self.num_nodes])
        
        # Define the observation space as the adjacency matrix of the network
        self.observation_space = spaces.Box(0, 1, shape=(self.num_nodes, self.num_nodes), dtype=int)
        
        # Return the initial state
        return self.get_state()
    
    def get_state(self):
        # Convert the MST to an adjacency matrix for the state representation
        adj_matrix = nx.to_numpy_array(self.tree, dtype=int)
        return adj_matrix
    
    def step(self, action):
        # Extract the parent and child nodes from the action
        parent, child = action

        # Initialize the reward and done flag
        reward = -10
        done = False

        # Check if the action is valid (parent is in the tree, child is not, and there is an edge in the network)
        if parent in self.tree and child not in self.tree and self.network.has_edge(parent, child):
            # Add the edge to the spanning tree
            self.tree.add_edge(parent, child)
            
            # Update the reward for a valid action
            reward = 1
            
            # Check if the spanning tree includes all nodes
            if len(self.tree) == self.num_nodes:
                done = True  # Episode ends when all nodes are added
        
        # Return the new state, reward, and done flag
        return self.get_state(), reward, done, {}

    def render(self, mode='human'):
        # Clear the previous plots
        self.ax[0].clear()
        self.ax[1].clear()
        
        # Draw the original physical network
        nx.draw(self.network, self.pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray', ax=self.ax[0])
        self.ax[0].set_title("Original Physical Network")
        
        # Draw the current spanning tree
        nx.draw(self.tree, self.pos, with_labels=True, node_color='lightgreen', node_size=700, edge_color='gray', ax=self.ax[1]) 
        self.ax[1].set_title("Spanning Tree")

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

# Example usage
if __name__ == "__main__":
    # Create the SpanningTreeEnv environment
    env = SpanningTreeEnv(min_nodes=5, max_nodes=15, min_redundancy=2, max_redundancy=4, show_weight_labels=True)
    
    # Reset the environment to start a new episode
    state = env.reset()
    done = False
    
    # Run the simulation loop until the episode is done
    while not done:
        # Select a random action from the action space
        action = env.action_space.sample()
        print(action)
        # Execute the action and get the new state, reward, and done flag
        state, reward, done, _ = env.step(action)
        
        # Render the current state of the environment
        env.render()
        
        # Update the Tkinter window
        env.root.update()
    
    print("Done")
    time.sleep(30)

    # Close the environment
    env.close()