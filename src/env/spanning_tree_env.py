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
                       num_attacked_nodes=2,
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
        self.num_attacked_nodes = num_attacked_nodes

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

        # Define action space as adjacency matrix
        # Since the action is symmetric, only define the upper triangular part
        # We only define the full matrix here, but ensure actions are taken on the upper triangular part through the environment logic
        # self.action_space = spaces.MultiBinary(self.max_nodes * self.max_nodes)

        # number of possible edges in an undirected graph without self-loops
        num_edges = int(self.num_nodes * (self.num_nodes - 1) / 2) 

        # Flat array representing only the upper triangle of the adjacency matrix.
        self.action_space = gym.spaces.MultiBinary(num_edges)

        # Define the observation space 
        # TODO currently observation space is max number of nodes. Explore embedding to an equal dimension.
        self.observation_space = spaces.Dict({
            "full_network": spaces.Box(low=0, high=1, shape=(self.num_nodes, self.num_nodes), dtype=np.int32),
            "mst": spaces.Box(low=0, high=1, shape=(self.num_nodes, self.num_nodes), dtype=np.int32),
            "weights": spaces.Box(low=0, high=10, shape=(self.num_nodes, self.num_nodes), dtype=np.int32),
            "attacked": spaces.Box(low=0, high=1, shape=(self.num_nodes, self.num_nodes), dtype=np.int32),
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
        weights_matrix_padded = np.zeros((size, size), dtype=float)
        weights_matrix_padded[:weights_matrix.shape[0], :weights_matrix.shape[1]] = weights_matrix

        # Create an adjacency matrix where attacked nodes are marked
        attacked_matrix = np.zeros((size, size), dtype=int)
        for node in self.attacked_nodes:
            # Mark entire row to indicate this node is attacked
            attacked_matrix[node, :] = 1  
            # Mark entire column to indicate this node is attacked
            attacked_matrix[:, node] = 1  

        # Complete State: Ensure all parts are padded to the maximum size
        return {
            "full_network": full_net_matrix_padded,
            "mst": mst_matrix_padded,
            "weights": weights_matrix_padded,
            "attacked": attacked_matrix,
        }

    def step(self, action):    

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

        # Render the current state of the environment if required
        if self.render_mode:
            self.render()
            self.root.update()

        return self.get_state(), reward, done, truncated, {}

    def execute_action(self, action):

        # Could be adding edge or removing edge
        valid_action = 0
        # Adding edge when physical network doesn't have edge
        invalid_action = 0
        # Added a connection to attacked node
        connected_to_attacked_node = 0 
        # Removed connection to attacked node
        disconnected_from_attacked_node = 0

        # Iterate over all possible connections 
        index = 0
        for i in range(self.num_nodes):
            # lower triangle not needed in full matrix approach
            for j in range(i + 1, self.num_nodes):
                # Check the action for the edge (i, j)
                if action[index] == 1 and not self.tree.has_edge(i, j):
                    if self.network.has_edge(i, j):
                        self.tree.add_edge(i, j, weight=self.network[i][j]['weight'])
                        valid_action += 1
                    else:
                        invalid_action += 1

                    if i in self.attacked_nodes or j in self.attacked_nodes:
                        connected_to_attacked_node += 1

                elif action[index] == 0 and self.tree.has_edge(i, j):
                    self.tree.remove_edge(i, j)
                    if i in self.attacked_nodes or j in self.attacked_nodes:
                        disconnected_from_attacked_node += 1
                index += 1

        return valid_action, invalid_action, connected_to_attacked_node, disconnected_from_attacked_node


    def is_connection_to_attacked_node(self, node1, node2):
        return node1 in self.attacked_nodes or node2 in self.attacked_nodes

    def calculate_reward(self, valid_action, invalid_action, connected_to_attacked_node, disconnected_from_attacked_node):
        
        reward = -.1
        done = False

        # Terminate episode if invalid action is taken
        if invalid_action >= 1:
            done = True 
            reward = -10
            return reward, done

        # # Apply penalties and rewards for actions related to attacked nodes
        # reward -= 1 * connected_to_attacked_node
        # reward += .5 * disconnected_from_attacked_node

        # # Check for connections to attacked nodes and penalize for each existing connection
        # for node in self.attacked_nodes:
        #     for neighbor in self.tree.neighbors(node):
        #         reward -= 0.5  # Penalize for each connection to an attacked node

        # Check if all attacked nodes are isolated
        all_isolated = self.is_attacked_isolated()
        if all_isolated:
            # reward += .5  # Reward for isolating attacked nodes
            non_attacked_subgraph = self.tree.subgraph([n for n in self.tree.nodes if n not in self.attacked_nodes])
            if nx.is_connected(non_attacked_subgraph) and nx.is_tree(non_attacked_subgraph):
                current_weight = sum(data['weight'] for u, v, data in non_attacked_subgraph.edges(data=True))
                reward += 50 - current_weight/100  # Encourage lighter tree
                done = True  # End the episode if a valid MST is formed
        #     else:
        #         reward -= .5  # Penalize if the subgraph is not a valid MST
        # else:
        #     reward -= .5  # Penalize if not all attacked nodes are isolated

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
        # Randomly select a few nodes to attack
        self.attacked_nodes = set(np.random.choice(self.network.nodes(), self.num_attacked_nodes, replace=False))
        
# Example usage
if __name__ == "__main__":
    # Create the SpanningTreeEnv environment
    env = SpanningTreeEnv(min_nodes=5, 
                          max_nodes=5, 
                          min_redundancy=3, 
                          max_redundancy=4, 
                          num_attacked_nodes=1,
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
        state, reward, done, _, _ = env.step(action)
        print(f' Action: \n {action} , Reward {reward}')
        time.sleep(.5)
        
        
        # Update the Tkinter window
        env.root.update()
    
    print("Done")
    time.sleep(30)

    # Close the environment
    env.close()