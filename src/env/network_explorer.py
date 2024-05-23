import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from network_env import NetworkEnvironment

MIN_REDUNDANCY = 2
MAX_REDUNDANCY = 4

def create_networks(sizes, count):
    """
    Generates multiple networks for each size and calculates their total weight.
    Args:
    sizes (list): A list of network sizes.
    count (int): Number of networks to generate per size.

    Returns:
    dict: A dictionary with keys as network sizes and values as lists of total weights.
    """
    network_weights = {size: [] for size in sizes}
    
    for size in sizes:
        for _ in range(count):
            # Create network environment with specific node count
            net_env = NetworkEnvironment(min_nodes=size, max_nodes=size, min_redundancy=MIN_REDUNDANCY, max_redundancy=MAX_REDUNDANCY)
            network = net_env.reset()  # Reset environment which also creates a new network

            # Calculate total weight of the network
            total_weight = sum(data['weight'] for _, _, data in network.edges(data=True))
            network_weights[size].append(total_weight)
    
    return network_weights

def plot_network_weights(network_weights):
    """
    Plots the network weights as a function of network size.
    Args:
    network_weights (dict): A dictionary with network sizes as keys and lists of weights as values.
    """
    plt.figure(figsize=(10, 6))
    
    # Prepare data for plotting
    sizes = sorted(network_weights.keys())
    weights_means = [np.mean(network_weights[size]) for size in sizes]
    weights_std = [np.std(network_weights[size]) for size in sizes]
    
    plt.errorbar(sizes, weights_means, yerr=weights_std, fmt='-o', capsize=5)
    plt.title('Network Total Weight vs. Size')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Average Total Weight')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    sizes = list(range(10, 150, 10))  # Network sizes from 10 to 150, every 10 steps
    networks = create_networks(sizes, 10)  # Create 10 networks for each size
    plot_network_weights(networks)  # Plot the results