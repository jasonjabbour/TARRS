import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import random
import numpy as np


def create_graph():
    community_size = 10
    community_num = 10
    p=0.01

    graph = nx.connected_caveman_graph(community_num, community_size)

    count = 0

    for (u, v) in graph.edges():
        if random.random() < p:  # rewire the edge
            x = random.choice(list(graph.nodes))
            if graph.has_edge(u, x):
                continue
            graph.remove_edge(u, v)
            graph.add_edge(u, x)
            count += 1
    print('rewire:', count)

    n = graph.number_of_nodes()
    label = np.zeros((n,n),dtype=int)
    for u in list(graph.nodes):
        for v in list(graph.nodes):
            if u//community_size == v//community_size and u>v:
                label[u,v] = 1
    rand_order = np.random.permutation(graph.number_of_nodes())
    feature = np.identity(graph.number_of_nodes())[:,rand_order]

    return graph

def select_malicious_nodes(tree, leader_node, num_malicious=1):
    malicious_nodes = []
    for _ in range(num_malicious):
        potential_malicious = [node for node in tree.nodes if node != leader_node and list(tree.successors(node))]
        if potential_malicious:
            malicious_node = random.choice(potential_malicious)
            tree.nodes[malicious_node]['malicious'] = True
            tree.nodes[malicious_node]['time'] = random.uniform(.5, 2)  # Assign an incorrect time
            tree.nodes[malicious_node]['impact'] = {child: random.uniform(.5, 100) for child in tree.successors(malicious_node)}
            malicious_nodes.append(malicious_node)
    return malicious_nodes

def assign_initial_clock_attributes(tree, leader_node):
    nx.set_node_attributes(tree, leader_node, 'leader')
    for node in tree.nodes:
        if node == leader_node:
            tree.nodes[node]['time'] = 0  # Leader clock starts at 0
            tree.nodes[node]['drift'] = random.uniform(0.00001, 0.0001)  # Very small drift
        else:
            path_length = nx.shortest_path_length(tree, source=leader_node, target=node)
            tree.nodes[node]['time'] = path_length * random.uniform(0.01, 1)  # Initial desynchronization
            tree.nodes[node]['drift'] = random.uniform(0.0001, 0.01)  # Random drift value
    return tree


def find_malicious_ancestor(node, tree, malicious_nodes):
    """Find the closest malicious ancestor of a node."""
    for ancestor in nx.ancestors(tree, node):
        if ancestor in malicious_nodes:
            return ancestor
    return None


def simulate_ptp_sync(tree, leader_node, malicious_nodes, sync_interval=5, steps=100, visualize_callback=None):
    for step in range(steps):
        for node in tree.nodes:
            if node != leader_node:
                parent = next(tree.predecessors(node), None)
                if parent:
                    # Check if the parent or any ancestor is malicious
                    malicious_ancestor = find_malicious_ancestor(parent, tree, malicious_nodes)
                    if malicious_ancestor:
                        # Apply the malicious impact for the child node (consistent across steps)
                        tree.nodes[node]['time'] = tree.nodes[malicious_ancestor]['time'] + tree.nodes[malicious_ancestor]['impact'].get(node, 0)
                    elif node not in malicious_nodes:
                        # Standard drift and sync with the parent
                        tree.nodes[node]['time'] += tree.nodes[node]['drift'] * sync_interval
                        
                        # Simulate delay and offset calculation
                        delay = random.uniform(0.0001, 0.005)
                        offset = tree.nodes[node]['time'] - (tree.nodes[parent]['time'] + delay)
                        
                        # Adjust time based on calculated offset
                        tree.nodes[node]['time'] -= offset / 2  # Adjust the offset so you don't converge immediately 
        
        # Call the visualization function if provided
        if visualize_callback:
            visualize_callback(tree, leader_node, malicious_nodes, step)


def visualize_sync(tree, leader_node, malicious_nodes, step, ax, pos, cmap, norm):
    ax.clear()

    # Node colors based on the current time
    node_colors = [
        'black' if node in malicious_nodes else ('cyan' if node == leader_node else cmap(norm(tree.nodes[node]['time'])))
        for node in tree.nodes
    ]
    
    # Draw the tree with updated colors
    nx.draw(tree, pos, with_labels=False, node_color=node_colors, edge_color='gray', node_size=1000, font_size=10, ax=ax)

    # Add node labels for time desynchronization and drift
    labels = {node: f"T: {tree.nodes[node]['time']:.2f}\nD: {tree.nodes[node]['drift']:.3f}" for node in tree.nodes}
    nx.draw_networkx_labels(tree, pos, labels=labels, font_size=8, font_color="black", ax=ax)

    ax.set_title(f"Step {step}: Clock Synchronization Visualization")
    plt.pause(.5)


if __name__ == '__main__':
    # Create graph
    graph = create_graph()

    # Create minimum spanning tree (MST) and convert it to a directed tree with a root
    tree = nx.minimum_spanning_tree(graph, weight='weight')
    leader_node = random.choice(list(tree.nodes))  # Optionally, select the leader node
    tree = nx.bfs_tree(tree, leader_node)  # Convert the MST to a directed tree (DAG) rooted at the leader node

    # Get initial clock attributes
    tree = assign_initial_clock_attributes(tree, leader_node)

    # Select malicious nodes
    malicious_nodes = select_malicious_nodes(tree, leader_node, num_malicious=2)

    # Get positions for the original graph
    pos = nx.spring_layout(graph)

    # Set up the plot
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(12, 12))

    # Set up colormap normalization without considering malicious nodes
    valid_times = [tree.nodes[node]['time'] for node in tree.nodes if node not in malicious_nodes]
    norm = mcolors.Normalize(vmin=min(valid_times), vmax=max(valid_times))
    cmap = plt.cm.Reds  # Use the 'Reds' colormap for heatmap effect

    # Simulate PTP synchronization and visualize the process live
    simulate_ptp_sync(tree, leader_node, malicious_nodes, sync_interval=5, steps=100,
                      visualize_callback=lambda t, l, m, s: visualize_sync(t, l, m, s, ax, pos, cmap, norm))

    plt.ioff()  # Disable interactive mode
    plt.show()