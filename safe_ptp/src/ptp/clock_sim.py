import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import random
import numpy as np


from safe_ptp.src.env.network_env import NetworkEnvironment

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
            tree.nodes[malicious_node]['time'] = random.uniform(1, 1000)  # Assign an incorrect time
            malicious_nodes.append(malicious_node)
    return malicious_nodes

def assign_initial_clock_attributes(tree, leader_node):
    nx.set_node_attributes(tree, leader_node, 'leader')
    for node in tree.nodes:
        tree.nodes[node]['has_malicious_ancestor'] = False  # Initialize the attribute
        if node == leader_node:
            tree.nodes[node]['time'] = 0  # Leader clock starts at 0
            tree.nodes[node]['drift'] = random.uniform(0.000001, 0.00001)  # Very small drift
            tree.nodes[node]['hops'] = 0
            tree.nodes[node]['type'] = 'leader'
        else:
            path_length = nx.shortest_path_length(tree, source=leader_node, target=node)
            tree.nodes[node]['hops'] = path_length
            tree.nodes[node]['time'] = random.uniform(0.1, 100)  # Initial desynchronization
            tree.nodes[node]['drift'] = random.uniform(0.0001, 0.01)  # Random drift value
            tree.nodes[node]['type'] = 'transparent'
        
    return tree

def assign_boundary_clocks(tree, leader_node, boundary_clock_ratio=0.1, hops_away_ratio=3):
    boundary_clocks = set()
    for node in tree.nodes:
        if node != leader_node and tree.nodes[node]['type'] == 'transparent':
            # Ensure the node has children (is not a leaf node)
            if random.random() < boundary_clock_ratio and tree.nodes[node]['hops'] >= hops_away_ratio and len(list(tree.successors(node))) > 0:
                tree.nodes[node]['type'] = 'boundary'
                # Make it pretty accurate
                tree.nodes[node]['drift'] = random.uniform(0.000001, 0.00001)  # Very small drift
                boundary_clocks.add(node)
    return boundary_clocks

def find_malicious_ancestor(node, tree, malicious_nodes):
    """Find the closest malicious ancestor of a node."""
    for ancestor in nx.ancestors(tree, node):
        if ancestor in malicious_nodes:
            return ancestor
    return None

def find_boundary_clock_ancestor(node, tree, boundary_clocks):
    """Find the closest boundary clock ancestor of a node."""
    for ancestor in nx.ancestors(tree, node):
        if ancestor in boundary_clocks:
            return ancestor
    return None
    

def simulate_ptp_sync(tree, leader_node, boundary_clocks, malicious_nodes, sync_interval=5, steps=100, visualize_callback=None):
    
    for step in range(steps):
        count = 0
        for node in tree.nodes:
            if node != leader_node:
                parent = list(tree.predecessors(node))
                if parent:
                    parent = parent[0]

                    # Check if the parent or any ancestor is malicious
                    malicious_ancestor = find_malicious_ancestor(node, tree, malicious_nodes)
                    if malicious_ancestor and (node not in malicious_nodes):
                        count+=1
                        # Apply the malicious impact for the child node (consistent across steps)
                        tree.nodes[node]['time'] = tree.nodes[malicious_ancestor]['time']
                        tree.nodes[node]['has_malicious_ancestor'] = True  # Mark the node as having a malicious ancestor
                    elif node not in malicious_nodes:
                        # Standard drift
                        tree.nodes[node]['time'] += tree.nodes[node]['drift'] * sync_interval
                        
                        # Determine which node to synchronize to
                        boundary_clock_ancestor = find_boundary_clock_ancestor(node, tree, boundary_clocks)
                        if boundary_clock_ancestor:
                            # Synchronize with boundary clock
                            sync_target = boundary_clock_ancestor
                            # If sychronizing to the boundary clock don't add full hop penality
                            hops = nx.shortest_path_length(tree, source=sync_target, target=node)
                        else:
                            # Synchronize with leader
                            sync_target = leader_node
                            # Hop to the leader node
                            hops = tree.nodes[node]['hops']

                        # Simulate delay
                        delay = random.uniform(0.01, 0.05) * hops
                        # Offset calculation
                        offset = tree.nodes[node]['time'] - (tree.nodes[sync_target]['time'] + delay)
                        
                        # Apply the offset correction
                        tree.nodes[node]['time'] -= offset / 2  # Simulate gradual synchronization

        # Call the visualization function if provided
        if visualize_callback:
            visualize_callback(tree, leader_node, boundary_clocks, malicious_nodes, step)


def visualize_sync(tree, leader_node, boundary_clocks, malicious_nodes, step, ax, pos, cmap, norm):
    ax.clear()

    # Separate nodes by type
    regular_nodes = [node for node in tree.nodes if node not in boundary_clocks and node not in malicious_nodes and not tree.nodes[node]['has_malicious_ancestor'] and node != leader_node]
    boundary_nodes = [node for node in boundary_clocks]
    leader_nodes = [leader_node]
    malicious_nodes_list = [node for node in malicious_nodes]
    affected_nodes = [node for node in tree.nodes if (tree.nodes[node]['has_malicious_ancestor'] and (node not in malicious_nodes))]

    # Draw edges
    nx.draw_networkx_edges(tree, pos, ax=ax, edge_color='grey', node_size=2000)

    # Draw regular nodes (circles)
    nx.draw_networkx_nodes(tree, pos, nodelist=regular_nodes, node_color=[cmap(norm(tree.nodes[node]['time'])) for node in regular_nodes],
                           node_size=1000, ax=ax, edgecolors='grey')

    # Add node labels for time desynchronization, drift, and hops
    labels = {node: f"T: {tree.nodes[node]['time']:.2f}\nD: {tree.nodes[node]['drift']:.3f}\nH:{tree.nodes[node]['hops']}" for node in tree.nodes}
    nx.draw_networkx_labels(tree, pos, labels=labels, font_size=8, font_color="black", ax=ax)

    # Draw leader node (cyan circle)
    nx.draw_networkx_nodes(tree, pos, nodelist=leader_nodes, node_color='cyan',
                           node_size=2000, ax=ax, edgecolors='black')

    # Draw malicious nodes (black circles with red outlines)
    nx.draw_networkx_nodes(tree, pos, nodelist=malicious_nodes_list, node_color='black',
                           node_size=1000, ax=ax, edgecolors='red')

    # Draw affected nodes (affected by malicious ancestors, with red outlines)
    nx.draw_networkx_nodes(tree, pos, nodelist=affected_nodes, node_color=[cmap(norm(tree.nodes[node]['time'])) for node in affected_nodes],
                           node_size=1000, ax=ax, edgecolors='red')

    # Draw boundary nodes (triangles)
    nx.draw_networkx_nodes(tree, pos, nodelist=boundary_nodes, node_color='orange',
                           node_size=1500, ax=ax, node_shape='^', edgecolors='black', alpha=.6)

    ax.set_title(f"Step {step}: Clock Synchronization Visualization")
    plt.pause(0.1)

if __name__ == '__main__':
    # Create graph
    graph = create_graph()

    # # Define the range for the number of nodes and redundancy
    # min_nodes = 100
    # max_nodes = 100
    # min_redundancy = 3

    # # Create the network environment with the specified parameters
    # network_env = NetworkEnvironment(min_nodes, max_nodes, min_redundancy)
    # # Generate and visualize the network
    # graph = network_env.reset()

    # Create minimum spanning tree (MST) and convert it to a directed tree with a root
    tree = nx.minimum_spanning_tree(graph, weight='weight')
    leader_node = random.choice(list(tree.nodes))  # Optionally, select the leader node
    tree = nx.bfs_tree(tree, leader_node)  # Convert the MST to a directed tree (DAG) rooted at the leader node

    # Get initial clock attributes
    tree = assign_initial_clock_attributes(tree, leader_node)

    # Assign boundary clocks
    boundary_clocks = assign_boundary_clocks(tree, leader_node)

    # Select malicious nodes
    malicious_nodes = select_malicious_nodes(tree, leader_node, num_malicious=2)

    # Get positions for the original graph
    pos = nx.spring_layout(graph)

    # Set up the plot
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(12, 12))

    # Set up colormap normalization without considering malicious nodes
    valid_times = [tree.nodes[node]['time'] for node in tree.nodes if node not in malicious_nodes]
    norm = mcolors.Normalize(vmin=0, vmax=10)
    cmap = plt.cm.Reds  # Use the 'Reds' colormap for heatmap effect

    # Simulate PTP synchronization and visualize the process live
    simulate_ptp_sync(tree, leader_node, boundary_clocks, malicious_nodes, sync_interval=5, steps=500,
                      visualize_callback=lambda t, l, b, m, s: visualize_sync(t, l, b, m, s, ax, pos, cmap, norm))

    plt.ioff()  # Disable interactive mode
    plt.show()


# ----------- More complex Implementation of PTP -----------
# def sync_message(master_time, delay):
#     return master_time + delay

# def follow_up_message(precise_master_time):
#     return precise_master_time  # Hardware timestamp of when the Sync message was actually sent

# def delay_request_response(slave_time, round_trip_delay):
#     return slave_time + round_trip_delay / 2

# def calculate_offset(T1, T2, T3, T4):
#     return ((T2 - T1) - (T4 - T3)) / 2

# def calculate_delay(T1, T2, T3, T4):
#     return ((T2 - T1) + (T4 - T3)) 

# def simulate_ptp_sync(tree, leader_node, malicious_nodes, sync_interval=5, steps=100, visualize_callback=None):
#     for step in range(steps):
#         # Sync Phase: Master sends Sync message
#         for node in tree.nodes:
#             if node != leader_node:
#                 parent = list(tree.predecessors(node))
#                 if parent:
#                     parent = parent[0]
#                     malicious_ancestor = find_malicious_ancestor(parent, tree, malicious_nodes)
#                     if malicious_ancestor and node not in malicious_nodes:
#                         tree.nodes[node]['time'] = tree.nodes[malicious_ancestor]['time'] + tree.nodes[malicious_ancestor]['impact'].get(node, 0)
#                         tree.nodes[node]['has_malicious_ancestor'] = True
#                     elif node not in malicious_nodes:
#                         # Update slave clock with its own drift
#                         tree.nodes[node]['time'] += tree.nodes[node]['drift'] * sync_interval
                        
#                         # Simulate delay for Sync and Follow-Up phases
#                         delay_master_to_slave = random.uniform(0.0001, 0.005) * tree.nodes[node]['hops']
#                         T1 = sync_message(tree.nodes[leader_node]['time'], delay_master_to_slave)
#                         precise_T1 = follow_up_message(tree.nodes[leader_node]['time'])  # Precise hardware timestamp
#                         T2 = tree.nodes[node]['time']  # Time when Sync message is received

#                         # Delay Request Phase: Slave sends delay request
#                         delay_slave_to_master = random.uniform(0.0001, 0.005) * tree.nodes[node]['hops']
#                         T3 = tree.nodes[node]['time']  # Time when Delay Request is sent
#                         T4 = delay_request_response(tree.nodes[leader_node]['time'], delay_master_to_slave + delay_slave_to_master)

#                         # Calculate the offset and delay
#                         offset = calculate_offset(precise_T1, T2, T3, T4)
#                         delay = calculate_delay(precise_T1, T2, T3, T4)

#                         # Adjust slave clock
#                         tree.nodes[node]['time'] -= offset  # Correct the offset
#                         tree.nodes[node]['time'] += delay / 2  # Compensate for delay

#         # Call the visualization function if provided
#         if visualize_callback:
#             visualize_callback(tree, leader_node, malicious_nodes, step)
