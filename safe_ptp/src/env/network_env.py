import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

LABEL_EDGES = False

class NetworkEnvironment:
    def __init__(self, min_nodes, max_nodes, min_redundancy):
        # Initialize the network environment with parameters and available generators
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.min_redundancy = min_redundancy
        self.num_nodes = None
        self.redundancy = None
        self.network = None
        self.positions = None

        # List of available graph generator function names
        self.generators = [
            'create_geometric_network',
            'create_watts_strogatz_network',
            'create_barabasi_albert_network',
            'create_erdos_renyi_graph',
            'create_powerlaw_cluster_graph',
            'create_ladder_graph',
            'create_star_graph',
            'create_wheel_graph'
        ]

    def get_generators(self):
        return self.generators
    
    def create_geometric_network(self, n, radius=0.3):
        """
        Create a random geometric graph.
        
        A geometric graph places nodes randomly in a metric space and 
        connects nodes that are within a certain distance (radius) from each other.
        
        :param n: The number of nodes.
        :param radius: The distance within which nodes are connected.
        """
        G = nx.random_geometric_graph(n, radius)
        return G, nx.get_node_attributes(G, 'pos')

    def create_watts_strogatz_network(self, n, k=4, p=0.1):
        """
        Create a Watts-Strogatz small-world graph.
        
        The Watts-Strogatz graph has high clustering and short average path lengths,
        mimicking social networks.
        
        :param n: The number of nodes.
        :param k: Each node is joined with its k nearest neighbors in a ring topology.
        :param p: The probability of rewiring each edge.
        """
        k = min(k, n - 1) 
        G = nx.watts_strogatz_graph(n, k, p)
        return G, nx.spring_layout(G)

    def create_barabasi_albert_network(self, n, m=2):
        """
        Create a Barabási-Albert scale-free network.
        
        The Barabási-Albert model generates scale-free networks using preferential attachment,
        where new nodes are more likely to connect to existing nodes with higher degrees.
        
        :param n: The number of nodes.
        :param m: The number of edges to attach from a new node to existing nodes.
        """
        G = nx.barabasi_albert_graph(n, m)
        return G, nx.spring_layout(G)

    def create_erdos_renyi_graph(self, n, p=0.1):
        """
        Create an Erdős-Rényi random graph.
        
        In an Erdős-Rényi graph, each pair of nodes is connected with a probability p,
        resulting in a binomial graph.
        
        :param n: The number of nodes.
        :param p: The probability of an edge between any two nodes.
        """
        G = nx.erdos_renyi_graph(n, p)
        return G, nx.spring_layout(G)

    def create_powerlaw_cluster_graph(self, n, m=2, p=0.1):
        """
        Create a Powerlaw Cluster graph.
        
        The Powerlaw Cluster graph is a modification of the Barabási-Albert model that 
        includes a mechanism for adding triangles, resulting in higher clustering.
        
        :param n: The number of nodes.
        :param m: The number of random edges to add for each new node.
        :param p: The probability of adding a triangle after adding a random edge.
        """
        G = nx.powerlaw_cluster_graph(n, m, p)
        return G, nx.spring_layout(G)
    
    def create_ladder_graph(self, n=10):
        """
        Create a ladder graph with n rungs.
        
        A ladder graph is a 2D grid graph with two parallel paths connected by edges,
        resembling the structure of a ladder.
        
        :param n: The number of rungs in the ladder.
        """
        G = nx.ladder_graph(n//2)
        return G, nx.spring_layout(G)

    def create_star_graph(self, n=10):
        """
        Create a star graph with n nodes.
        
        A star graph consists of one central node connected to all other nodes, 
        resembling a star.
        
        :param n: The total number of nodes in the star.
        """
        G = nx.star_graph(n-1)
        return G, nx.spring_layout(G)

    def create_wheel_graph(self, n=10):
        """
        Create a wheel graph with n nodes.
        
        A wheel graph consists of a central hub node connected to all nodes in a cycle.
        
        :param n: The total number of nodes in the wheel.
        """
        G = nx.wheel_graph(n)
        return G, nx.spring_layout(G)
    

    def create_combined_network(self, n):
        """
        Create a network by combining two randomly selected graph generators.
        
        :param n: The number of nodes.
        """
        gen1, gen2 = random.sample(self.generators, 2)
        # print(f'Combining {gen1} and {gen2}')
        
        # Determine the split of nodes
        n1 = n // 2
        n2 = n - n1
        
        # Get the generator functions
        generator1 = getattr(self, gen1)
        generator2 = getattr(self, gen2)
        
        # Generate two subgraphs
        G1, pos1 = generator1(n1)
        G2, pos2 = generator2(n2)
        
        # Relabel the nodes of G2 to avoid conflicts with G1
        G2 = nx.relabel_nodes(G2, {i: i + n1 for i in range(n2)})
        pos2 = {i + n1: pos for i, pos in pos2.items()}
        
        # Combine the graphs
        G_combined = nx.compose(G1, G2)
        pos_combined = {**pos1, **pos2}
        
        # Remove self-loops in the combined graph
        self_loops = list(nx.selfloop_edges(G_combined))
        G_combined.remove_edges_from(self_loops)
        
        # Connect G1 and G2 with random edges
        nodes_G1 = list(G1.nodes())
        nodes_G2 = list(G2.nodes())
        num_edges_to_add = min(len(nodes_G1), len(nodes_G2), 3)  # Adding 3 edges or less if nodes are fewer
        
        for _ in range(num_edges_to_add):
            u = random.choice(nodes_G1)
            v = random.choice(nodes_G2)
            # Ensure no self-loops and no duplicate edges
            while u == v or G_combined.has_edge(u, v):
                u = random.choice(nodes_G1)
                v = random.choice(nodes_G2)
            G_combined.add_edge(u, v)
        
        return G_combined, pos_combined
        
    def create_redundant_network(self, generator, *args, **kwargs):
        """
        Create a network using a specified generator function.
        The generator function should return a NetworkX graph and positions.
        """

        # If unconnected graph is generated, try again
        while True:
            # Generate the base graph using the specified generator
            G, positions = generator(*args, **kwargs)
            # Break out once a you get a valid graph
            if nx.is_connected(G):
                break
                
        # Ensure all nodes have at least min_redundancy connections
        for node in G.nodes():
            while G.degree[node] < self.min_redundancy:
                target = np.random.randint(0, len(G.nodes()))
                if node != target and not G.has_edge(node, target):
                    G.add_edge(node, target)
        
        return G, positions

    def assign_edge_weights(self, positions):
        # Assign weights to edges based on Euclidean distances derived from positions
        for u, v in self.network.edges():
            weight = np.linalg.norm(np.array(positions[u]) - np.array(positions[v]))
            self.network[u][v]['weight'] = round(weight, 2)

    def reset(self):
        """
        Reset the environment by clearing the existing network and creating a new one
        using a randomly selected generator function, including position-based weights.
        """
        # How often you get combined networks
        com_net_occurancy_rate = 5 

        while True:
            try:
                if self.network is not None:
                    self.network.clear()
                    self.positions = None

                # Randomly select a generator function from the list of available generators
                generator_name = random.choice(self.generators + ['create_combined_network'] * com_net_occurancy_rate)
                # print(f'Generator: {generator_name}')
                generator = getattr(self, generator_name)

                # Randomly determine the number of nodes
                self.num_nodes = np.random.randint(self.min_nodes, self.max_nodes + 1)

                # Create the network using the selected generator
                self.network, self.positions = self.create_redundant_network(generator, self.num_nodes)

                # Assign weights to the edges
                self.assign_edge_weights(self.positions)
                
                return self.network
            except Exception as e:
                # print(f"Error in creating network: {e}. Retrying...")
                pass
    
    def visualize_network(self, G, pos, add_edge_labels=False, title="Network Visualization"):
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
    
    def get_positions(self):
        return self.positions
    
    def print_node_degrees(self):
        """
        Print the degrees of each node in the network.
        """
        for node, degree in self.network.degree():
            print(f"Node {node}: Degree {degree}")


if __name__ == '__main__':
    # Define the range for the number of nodes and redundancy
    min_nodes = 50
    max_nodes = 50
    min_redundancy = 3

    # Create the network environment with the specified parameters
    network_env = NetworkEnvironment(min_nodes, max_nodes, min_redundancy)
    
    # Generate and visualize the network
    network = network_env.reset()
            
    # network_env.print_node_degrees()
    network_env.visualize_network(network, network_env.positions, add_edge_labels=LABEL_EDGES, title="Random Network Visualization")
