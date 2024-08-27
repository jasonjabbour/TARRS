import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation
import random
import numpy as np

from safe_ptp.src.env.network_env import NetworkEnvironment

class ClockSimulation:
    def __init__(self, community_size=10, community_num=10):
        self.community_size = community_size
        self.community_num = community_num
        self.graph = None
        self.tree = None
        self.leader_node = None
        self.boundary_clocks = set()
        self.malicious_nodes = []

        # Create graph
        self.create_graph()

        # # Define the range for the number of nodes and redundancy
        # min_nodes = 100
        # max_nodes = 100
        # min_redundancy = 3

        # # Create the network environment with the specified parameters
        # network_env = NetworkEnvironment(min_nodes, max_nodes, min_redundancy)
        # # Generate and visualize the network
        # self.graph = network_env.reset()

        # Choose random leader node
        self.leader_node = random.choice(list(self.graph.nodes))

        # Assign initial clock attributes to nodes
        self.assign_initial_clock_attributes()

        # Build initial tree
        self.reconfigure_tree()

        # Assign the boundary clocks
        self.assign_boundary_clocks()

        # Assign the malicious nodes
        self.select_malicious_nodes(num_malicious=2)

        self.randomly_disconnect_nodes()

    # TODO: Move this to Network Environment Class
    def create_graph(self):
        """Create the initial graph."""
        graph = nx.connected_caveman_graph(self.community_num, self.community_size)
        count = 0
        p = 0.01

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
        label = np.zeros((n, n), dtype=int)
        for u in list(graph.nodes):
            for v in list(graph.nodes):
                if u // self.community_size == v // self.community_size and u > v:
                    label[u, v] = 1
        rand_order = np.random.permutation(graph.number_of_nodes())
        feature = np.identity(graph.number_of_nodes())[:, rand_order]

        self.graph = graph

    def reconfigure_tree(self, randomize_weights=False):
        """Reconfigure the tree structure, preserving node states."""

        # Option to randomize weights to get a different spanning tree
        if randomize_weights:
            for u, v, d in self.graph.edges(data=True):
                d['weight'] = random.uniform(0.1, 1.0)  # Randomize weights between 0.1 and 1.0

        # Create minimum spanning tree (MST) and convert it to a directed tree with a root
        self.tree = nx.minimum_spanning_tree(self.graph, weight='weight')
        self.tree = nx.bfs_tree(self.tree, self.leader_node)  # Convert the MST to a directed tree (DAG) rooted at the leader node

        # Copy node attributes from the original graph to the tree
        for node in self.tree.nodes:
            self.tree.nodes[node].update(self.graph.nodes[node])

        # Assign tree specific node features
        for node in self.tree.nodes:
            path_length = nx.shortest_path_length(self.tree, source=self.leader_node, target=node)
            # Assign to both graph and tree
            self.graph.nodes[node]['hops'] = path_length 
            self.tree.nodes[node]['hops'] = path_length 

    def randomly_disconnect_nodes(self, num_disconnections=1):
        # Select nodes to disconnect
        nodes_to_disconnect = random.sample(list(self.tree.nodes), num_disconnections)

        for node in nodes_to_disconnect:
            # Find the parent of the node (if any)
            parents = list(self.tree.predecessors(node))
            
            # Disconnect the node from its parent
            for parent in parents:
                self.tree.remove_edge(parent, node)
            
            # Label the node as disconnected
            self.tree.nodes[node]['disconnected'] = True
            self.graph.nodes[node]['disconnected'] = True

    def select_malicious_nodes(self, num_malicious=1):
        """Select and configure malicious nodes."""
        malicious_nodes = []
        for _ in range(num_malicious):
            potential_malicious = [node for node in self.tree.nodes if node != self.leader_node and list(self.tree.successors(node))]
            if potential_malicious:
                malicious_node = random.choice(potential_malicious)
                # Set malicious attribute for both tree and graph node
                self.tree.nodes[malicious_node]['is_malicious'] = True
                self.graph.nodes[malicious_node]['is_malicious'] = True

                # Set bad time for malicious node in both graph and tree
                time = random.uniform(1, 1000)  # Assign an incorrect time
                self.tree.nodes[malicious_node]['time'] = time
                self.graph.nodes[malicious_node]['time'] = time
                malicious_nodes.append(malicious_node)
        self.malicious_nodes = malicious_nodes

    def assign_initial_clock_attributes(self):
        """Assign initial clock attributes to nodes."""
        nx.set_node_attributes(self.graph, self.leader_node, 'leader')
        for node in self.graph.nodes:
            # Initialize the attribute for all nodes
            self.graph.nodes[node]['has_malicious_ancestor'] = False
            self.graph.nodes[node]['is_malicious'] = False
            self.graph.nodes[node]['disconnected'] = False
            # Susceptibility to attack ratio that will determine how bad an attack influences this node
            self.graph.nodes[node]['susceptibility'] = random.uniform(0.1, 10) + random.uniform(1, 1000) * random.randint(0, 1)
            if node == self.leader_node:
                self.graph.nodes[node]['time'] = 0  # Leader clock starts at 0
                self.graph.nodes[node]['drift'] = random.uniform(0.000001, 0.00001)  # Very small drift
                self.graph.nodes[node]['hops'] = 0
                self.graph.nodes[node]['type'] = 'leader'
            else:
                self.graph.nodes[node]['hops'] = None # Tree has not been constructed yet
                self.graph.nodes[node]['time'] = random.uniform(0.1, 100)  # Initial desynchronization
                self.graph.nodes[node]['drift'] = random.uniform(0.0001, 0.01)  # Random drift value
                self.graph.nodes[node]['type'] = 'transparent'
 
    def assign_boundary_clocks(self, boundary_clock_ratio=0.5, hops_away_ratio=3):
        """Assign boundary clocks based on specific criteria."""
        boundary_clocks = set()
        for node in self.tree.nodes:
            if node != self.leader_node and self.tree.nodes[node]['type'] == 'transparent':
                # Ensure the node has children (is not a leaf node)
                if random.random() < boundary_clock_ratio and self.tree.nodes[node]['hops'] >= hops_away_ratio and len(list(self.tree.successors(node))) > 0:
                    self.tree.nodes[node]['type'] = 'boundary'
                    self.graph.nodes[node]['type'] = 'boundary'
                    # Make clock pretty accurate
                    drift = random.uniform(0.000001, 0.00001) 
                    self.tree.nodes[node]['drift'] = drift
                    self.graph.nodes[node]['drift'] = drift
                    boundary_clocks.add(node)
        self.boundary_clocks = boundary_clocks

    def find_malicious_ancestor(self, node):
        """Find the closest malicious ancestor of a node."""
        for ancestor in nx.ancestors(self.tree, node):
            if ancestor in self.malicious_nodes:
                return ancestor
        return None

    def find_boundary_clock_ancestor(self, node):
        """Find the closest boundary clock ancestor of a node."""
        for ancestor in nx.ancestors(self.tree, node):
            if ancestor in self.boundary_clocks:
                return ancestor
        return None
    
    def is_boundary_clock_below_malicious(self, malicious_ancestor, boundary_ancestor):
        """Check if the boundary clock ancestor is below the malicious ancestor."""
        if malicious_ancestor and boundary_ancestor:
            # Check if the boundary clock ancestor is a descendant of the malicious ancestor
            if nx.has_path(self.tree, malicious_ancestor, boundary_ancestor):
                return True
        return False
    
    def simulate_ptp_sync(self, sync_interval=5, steps=100, visualize_callback=None):
        """Simulate PTP synchronization across the network."""
        for step in range(steps):
            for node in self.tree.nodes:
                if node != self.leader_node:
                    parent = list(self.tree.predecessors(node))
                    if len(parent) > 0:
                        parent = parent[0]
                        # If parent exists that means node is not disconnected
                        self.graph.nodes[node]['disconnected'] = False
                        self.tree.nodes[node]['disconnected'] = False

                        # Check if the parent or any ancestor is malicious
                        malicious_ancestor = self.find_malicious_ancestor(node)
                        # Determine which node to synchronize to
                        boundary_clock_ancestor = self.find_boundary_clock_ancestor(node)
                        # See if the boundary clock is closer ancestor than malicious ancestor
                        is_boundary_closer_ancestor = self.is_boundary_clock_below_malicious(malicious_ancestor, boundary_clock_ancestor)

                        # Label the node as attacked
                        if malicious_ancestor and (node not in self.malicious_nodes):
                            self.tree.nodes[node]['has_malicious_ancestor'] = True
                            self.graph.nodes[node]['has_malicious_ancestor'] = True
                        else:
                            self.tree.nodes[node]['has_malicious_ancestor'] = False
                            self.graph.nodes[node]['has_malicious_ancestor'] = False

                        # Decide if attack impact should come from attacker or boundary clock
                        if self.tree.nodes[node]['has_malicious_ancestor'] and not is_boundary_closer_ancestor:
                            # Apply the malicious impact for the child node (consistent across steps)
                            self.tree.nodes[node]['time'] = self.tree.nodes[node]['susceptibility']
                            self.graph.nodes[node]['time'] = self.tree.nodes[node]['susceptibility']

                        elif node not in self.malicious_nodes:
                            # Standard drift
                            # TODO: Should visualize the drift then synchronize
                            self.simulate_drift(node, sync_interval)

                            if boundary_clock_ancestor:
                                # Synchronize with boundary clock
                                sync_target = boundary_clock_ancestor
                                # If synchronizing to the boundary clock don't add full hop penalty
                                hops = nx.shortest_path_length(self.tree, source=sync_target, target=node)
                            else:
                                # Synchronize with leader
                                sync_target = self.leader_node
                                # Hop to the leader node
                                hops = self.tree.nodes[node]['hops']

                            # Synchronize node
                            self.sync_clock(node, sync_target, hops)

                    else:
                        # No parent means disconnected
                        self.graph.nodes[node]['disconnected'] = True
                        self.tree.nodes[node]['disconnected'] = True

                        # Let node drift away
                        self.simulate_drift(node, sync_interval, scaling_ratio=1000)

            # Call the visualization function if provided
            if visualize_callback:
                visualize_callback(self.tree, self.leader_node, self.boundary_clocks, self.malicious_nodes, step)

    def sync_clock(self, node, sync_target, hops):
        # Simulate delay
        delay = random.uniform(0.01, 0.05) * hops
        # Offset calculation
        offset = self.tree.nodes[node]['time'] - (self.tree.nodes[sync_target]['time'] + delay)

        # Apply the offset correction
        self.tree.nodes[node]['time'] -= offset / 2  # Simulate gradual synchronization
        self.graph.nodes[node]['time'] -= offset / 2  # Simulate gradual synchronization
    
    def simulate_drift(self, node, sync_interval, scaling_ratio=1):
        self.tree.nodes[node]['time'] += self.tree.nodes[node]['drift'] * sync_interval * scaling_ratio
        self.graph.nodes[node]['time'] += self.tree.nodes[node]['drift'] * sync_interval * scaling_ratio

    def get_total_desync_time(self):
        """Get the total time across all nodes in the tree."""
        total_desync_time = sum(self.tree.nodes[node]['time'] for node in self.tree.nodes)
        return total_desync_time
    
    def visualize_sync(self, step, ax, pos, cmap, norm):
        """Visualize the current state of synchronization."""
        ax.clear()

        # Separate nodes by type
        regular_nodes = [node for node in self.tree.nodes if node not in self.boundary_clocks and node not in self.malicious_nodes and not self.tree.nodes[node]['has_malicious_ancestor'] and node != self.leader_node]
        boundary_nodes = [node for node in self.boundary_clocks]
        leader_nodes = [self.leader_node]
        malicious_nodes_list = [node for node in self.malicious_nodes]
        affected_nodes = [node for node in self.tree.nodes if (self.tree.nodes[node]['has_malicious_ancestor'] and (node not in self.malicious_nodes))]
        disconnected_nodes = [node for node in self.tree.nodes if self.tree.nodes[node]['disconnected']]

        # Draw graph edges in light grey
        nx.draw_networkx_edges(self.graph, pos, ax=ax, edge_color='lightgrey', width=1.0)

        # Draw tree edges in black on top of the graph edges
        nx.draw_networkx_edges(self.tree, pos, ax=ax, edge_color='black', width=1.0, node_size=2000)

        # Draw regular nodes (circles)
        nx.draw_networkx_nodes(self.tree, pos, nodelist=regular_nodes, node_color=[cmap(norm(self.tree.nodes[node]['time'])) for node in regular_nodes],
                               node_size=1000, ax=ax, edgecolors='grey')

        # Add node labels for time desynchronization, drift, and hops
        labels = {node: f"T: {self.tree.nodes[node]['time']:.2f}\nD: {self.tree.nodes[node]['drift']:.3f}\nH:{self.tree.nodes[node]['hops']}" for node in self.tree.nodes}
        nx.draw_networkx_labels(self.tree, pos, labels=labels, font_size=8, font_color="black", ax=ax)

        # Draw leader node (cyan circle)
        nx.draw_networkx_nodes(self.tree, pos, nodelist=leader_nodes, node_color='cyan',
                               node_size=2000, ax=ax, edgecolors='black')

        # Draw malicious nodes (black circles with red outlines)
        nx.draw_networkx_nodes(self.tree, pos, nodelist=malicious_nodes_list, node_color='black',
                               node_size=1000, ax=ax, edgecolors='red')

        # Draw affected nodes (affected by malicious ancestors, with red outlines)
        nx.draw_networkx_nodes(self.tree, pos, nodelist=affected_nodes, node_color=[cmap(norm(self.tree.nodes[node]['time'])) for node in affected_nodes],
                               node_size=1000, ax=ax, edgecolors='red')

        # Draw boundary nodes (triangles)
        nx.draw_networkx_nodes(self.tree, pos, nodelist=boundary_nodes, node_color='orange',
                               node_size=1500, ax=ax, node_shape='^', edgecolors='black', alpha=.6)
        
        # Draw disconnected nodes
        nx.draw_networkx_nodes(self.tree, pos, nodelist=disconnected_nodes, node_color='black',
                               node_size=2000, ax=ax, node_shape='p', alpha=.3)

        ax.set_title(f"Step {step}: Clock Synchronization Visualization")
        plt.pause(0.1)

# Main execution
if __name__ == '__main__':

    # Create a PTP Simulation instance
    clock_sim = ClockSimulation()

    # Get positions for the original graph
    pos = nx.spring_layout(clock_sim.graph)

    # Set up the plot
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(12, 12))

    # Set up colormap normalization without considering malicious nodes
    norm = mcolors.Normalize(vmin=0, vmax=10)
    cmap = plt.cm.Reds  # Use the 'Reds' colormap for heatmap effect

    # Simulate PTP synchronization and visualize the process live
    clock_sim.simulate_ptp_sync(sync_interval=5, steps=50,
                              visualize_callback=lambda t, l, b, m, s: clock_sim.visualize_sync(s, ax, pos, cmap, norm))

    # You can also reconfigure the graph
    num_reconfigurations = 10
    for _ in range(num_reconfigurations):
        # Reconfigure the tree (keeping the same nodes and states)
        clock_sim.reconfigure_tree(randomize_weights=True)

        # Select node to disconnect 
        clock_sim.randomly_disconnect_nodes()

        # Simulate PTP synchronization and visualize the process live after reconfiguration
        clock_sim.simulate_ptp_sync(sync_interval=5, steps=50,
                                visualize_callback=lambda t, l, b, m, s: clock_sim.visualize_sync(s, ax, pos, cmap, norm))
        
    # Disable interactive mode
    plt.ioff()  
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
