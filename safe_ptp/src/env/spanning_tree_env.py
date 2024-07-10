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
                       max_redundancy, 
                       start_difficulty_level=1,
                       final_difficulty_level=10,
                       num_timestep_cooldown=10000, 
                       min_attacked_nodes=1,
                       max_attacked_nodes=2,
                       show_weight_labels=False, 
                       render_mode=False, 
                       max_ep_steps=100, 
                       node_size=700, 
                       history_size=500, 
                       performance_threshold=30):
        super(SpanningTreeEnv, self).__init__()

        # Initialize parameters for the network environment
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.min_redundancy = min_redundancy
        self.max_redundancy = max_redundancy
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
       
        # Initialize placeholders for the number of nodes, action space, and observation space
        self.max_difficulty_num_nodes = 4 + NUM_NODE_INCREASE_RATE_PER_LEVEL *  self.final_difficulty_level

        # Define action space as adjacency matrix
        # Since the action is symmetric, only define the upper triangular part
        # We only define the full matrix here, but ensure actions are taken on the upper triangular part through the environment logic
        # self.action_space = spaces.MultiBinary(self.max_nodes * self.max_nodes)

        # number of possible edges in an undirected graph without self-loops
        self.max_difficulty_max_num_edges = int(self.max_difficulty_num_nodes * (self.max_difficulty_num_nodes - 1) / 2) 

        # Flat array representing only the upper triangle of the adjacency matrix. (FOR PPO)
        self.action_space = spaces.MultiBinary(self.max_difficulty_max_num_edges)

        # Define a continuous action space where each action can range from 0 to 1 (FOR SAC)
        # self.action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(self.max_difficulty_max_num_edges,), dtype=np.float32)

        # Define the observation space 
        # TODO currently observation space is max number of nodes. Explore embedding to an equal dimension.
        self.observation_space = spaces.Dict({
            "full_network": spaces.Box(low=0, high=1, shape=(self.max_difficulty_num_nodes, self.max_difficulty_num_nodes), dtype=np.int32),
            "mst": spaces.Box(low=0, high=1, shape=(self.max_difficulty_num_nodes, self.max_difficulty_num_nodes), dtype=np.int32),
            "weights": spaces.Box(low=0, high=10, shape=(self.max_difficulty_num_nodes, self.max_difficulty_num_nodes), dtype=np.float32),
            "attacked": spaces.MultiBinary(self.max_difficulty_num_nodes),
            "action_mask": spaces.MultiBinary(self.max_difficulty_max_num_edges)
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
        # Keep track of number of nodes in each env
        self.num_nodes_history.append(self.num_nodes)

        # Simulate attack
        self.simulate_attack()

        # Calculate max number of edges for this network
        self.current_network_max_num_edges = int(self.num_nodes * (self.num_nodes - 1) / 2) 

        # Return the initial state
        return self.get_state(), {}

    def get_state(self):

        # Function to pad matrices to the maximum node size
        size = self.max_difficulty_num_nodes

        # Convert the full network and MST to adjacency matrices
        full_net_matrix = nx.to_numpy_array(self.network, weight=None, dtype=int) # WEIGHT MUST BE SET TO NONE!
        mst_matrix = nx.to_numpy_array(self.tree, weight=None, dtype=int)         # WEIGHT MUST BE SET TO NONE!

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
        attacked_vector = np.zeros(size, dtype=int)
        for node in self.attacked_nodes:
            attacked_vector[node] = 1

        # Get action mask
        action_mask = self.get_action_mask()

        # Complete State: Ensure all parts are padded to the maximum size
        return {
            "full_network": full_net_matrix_padded,
            "mst": mst_matrix_padded,
            "weights": weights_matrix_padded,
            "attacked": attacked_vector,
            "action_mask": action_mask,
        }

    def get_action_mask(self):
        
        # Initialize mask with zeros
        action_mask = np.zeros(self.action_space.shape[0], dtype=int)
        # Activate the mask for the valid actions based on 
        # number of valid edges (actions) for the current number of active nodes
        action_mask[:self.current_network_max_num_edges] = 1
        
        return action_mask

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

        # Could be adding edge or removing edge
        valid_action = 0
        # Adding edge when physical network doesn't have edge
        invalid_action = 0
        # Added a connection to attacked node
        connected_to_attacked_node = 0 
        # Removed connection to attacked node
        disconnected_from_attacked_node = 0

        # Iterate over all possible connections within the current network size 
        index = 0
        # Iterate over only the number of nodes in this current network size
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

        # # Terminate episode if invalid action is taken
        # if invalid_action >= 1:
        #     done = True 
        #     reward = -10
        #     return reward, done

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
            reward += .5  # Reward for isolating attacked nodes
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
                          max_redundancy=4, 
                          min_attacked_nodes=1, 
                          max_attacked_nodes=2,
                          start_difficulty_level=1,
                          final_difficulty_level=5,
                          num_timestep_cooldown=2, 
                          show_weight_labels=SHOW_WEIGHT_LABELS, 
                          render_mode=True, 
                          max_ep_steps=3, 
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
        # print(state['full_network'])
        time.sleep(.5)
        
        # Update the Tkinter window
        env.root.update()

        if done:
            state = env.reset()
            done = False
    
    print("Done")
    time.sleep(30)

    # Close the environment
    env.close()