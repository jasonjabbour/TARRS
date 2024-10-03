        # # Initialize placeholders for the number of nodes, action space, and observation space
        # self.max_difficulty_num_nodes = 4 + NUM_NODE_INCREASE_RATE_PER_LEVEL *  self.final_difficulty_level

        # # number of possible edges in an undirected graph without self-loops
        # self.max_difficulty_max_num_edges = int(self.max_difficulty_num_nodes * (self.max_difficulty_num_nodes - 1) / 2) 

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