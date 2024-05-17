import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

LABEL_EDGES = True

class NetworkEnvironment:
    def __init__(self, min_nodes, max_nodes, min_redundancy, max_redundancy):
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.min_redundancy = min_redundancy
        self.max_redundancy = max_redundancy
        self.num_nodes = None
        self.redundancy = None
        self.network = None

    def create_redundant_network(self, n, redundancy=2):
        """
        Create a redundant network with `n` nodes and `redundancy` edges per node.
        """
        G = nx.Graph()
        
        # Add nodes to the graph
        G.add_nodes_from(range(n))
        
        # Randomly connect nodes with the specified redundancy
        for i in range(n):
            # Ensure each node has at least `redundancy` connections
            while G.degree[i] < redundancy:
                j = np.random.randint(0, n)
                if i != j:
                    G.add_edge(i, j)
        
        return G

    def get_positions(self, G):
        return nx.spring_layout(G, seed=42)

    def assign_edge_weights(self):
        # Assign weights to edges based on Euclidean distances derived from positions
        for u, v in self.network.edges():
            weight = np.linalg.norm(np.array(self.positions[u]) - np.array(self.positions[v]))
            self.network[u][v]['weight'] = round(weight, 2)

    def reset(self):
        """
        Reset the environment by clearing the existing network and creating a new one
        with a random number of nodes and redundancy, including position-based weights.
        """
        if self.network is not None:
            self.network.clear()

        self.num_nodes = np.random.randint(self.min_nodes, self.max_nodes + 1)
        self.redundancy = np.random.randint(self.min_redundancy, self.max_redundancy + 1)
        self.network = self.create_redundant_network(self.num_nodes, self.redundancy)
        self.positions = self.get_positions(self.network)
        self.assign_edge_weights()
        return self.network
    
    def visualize_network(self, G, pos, add_edge_labels=False, title="Redundant Network Visualization"):
        """
        Visualize the network using matplotlib, showing the nodes, weighted edges,
        and using the positions that determined those weights.
        """
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray', width=1)

        if add_edge_labels:
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            
        plt.title(title)
        plt.show()


if __name__ == '__main__':

    min_nodes = 5
    max_nodes = 30
    min_redundancy = 3
    max_redundancy = 5

    # Create and visualize the network
    network_env = NetworkEnvironment(min_nodes, max_nodes, min_redundancy, max_redundancy)
    network = network_env.reset()
    network_env.visualize_network(network, network_env.positions, add_edge_labels=LABEL_EDGES, title="Example Network Visualization")
