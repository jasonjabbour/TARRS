import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


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

    def reset(self):
        # Explicitly clear the previous network
        if self.network is not None:
            self.network.clear()

        self.num_nodes = np.random.randint(self.min_nodes, self.max_nodes + 1)
        self.redundancy = np.random.randint(self.min_redundancy, self.max_redundancy + 1)
        self.network = self.create_redundant_network(self.num_nodes, self.redundancy)
        return self.network
    
    def visualize_network(self, G, title="Redundant Network Visualization"):
        """
        Visualize the network using matplotlib.
        """
        pos = nx.spring_layout(G)
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray')
        plt.title(title)
        plt.show()

if __name__ == '__main__':

    min_nodes = 10
    max_nodes = 100
    min_redundancy = 3
    max_redundancy = 5

    # Create and visualize the network
    network_env = NetworkEnvironment(min_nodes, max_nodes, min_redundancy, max_redundancy)
    network = network_env.reset()
    network_env.visualize_network(network)
