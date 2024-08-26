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

    def reconfigure_tree(self):
        """Reconfigure the tree structure, preserving node states."""
        # Create minimum spanning tree (MST) and convert it to a directed tree with a root
        self.tree = nx.minimum_spanning_tree(self.graph, weight='weight')

        if self.leader_node is None:
            self.leader_node = random.choice(list(self.tree.nodes)) 

        self.tree = nx.bfs_tree(self.tree, self.leader_node)  # Convert the MST to a directed tree (DAG) rooted at the leader node

    def select_malicious_nodes(self, num_malicious=1):
        """Select and configure malicious nodes."""
        malicious_nodes = []
        for _ in range(num_malicious):
            potential_malicious = [node for node in self.tree.nodes if node != self.leader_node and list(self.tree.successors(node))]
            if potential_malicious:
                malicious_node = random.choice(potential_malicious)
                self.tree.nodes[malicious_node]['is_malicious'] = True
                self.tree.nodes[malicious_node]['time'] = random.uniform(1, 1000)  # Assign an incorrect time
                malicious_nodes.append(malicious_node)
        self.malicious_nodes = malicious_nodes

    def assign_initial_clock_attributes(self):
        """Assign initial clock attributes to nodes."""
        nx.set_node_attributes(self.tree, self.leader_node, 'leader')
        for node in self.tree.nodes:
            # Initialize the attribute for all nodes
            self.tree.nodes[node]['has_malicious_ancestor'] = False
            self.tree.nodes[node]['is_malicious'] = False
            # Susceptibility to attack ratio that will determine how bad an attack influences this node
            self.tree.nodes[node]['susceptibility'] = random.uniform(0.1, 10) + random.uniform(1, 1000) * random.randint(0, 1)
            if node == self.leader_node:
                self.tree.nodes[node]['time'] = 0  # Leader clock starts at 0
                self.tree.nodes[node]['drift'] = random.uniform(0.000001, 0.00001)  # Very small drift
                self.tree.nodes[node]['hops'] = 0
                self.tree.nodes[node]['type'] = 'leader'
            else:
                path_length = nx.shortest_path_length(self.tree, source=self.leader_node, target=node)
                self.tree.nodes[node]['hops'] = path_length
                self.tree.nodes[node]['time'] = random.uniform(0.1, 100)  # Initial desynchronization
                self.tree.nodes[node]['drift'] = random.uniform(0.0001, 0.01)  # Random drift value
                self.tree.nodes[node]['type'] = 'transparent'
 
    def assign_boundary_clocks(self, boundary_clock_ratio=0.5, hops_away_ratio=3):
        """Assign boundary clocks based on specific criteria."""
        boundary_clocks = set()
        for node in self.tree.nodes:
            if node != self.leader_node and self.tree.nodes[node]['type'] == 'transparent':
                # Ensure the node has children (is not a leaf node)
                if random.random() < boundary_clock_ratio and self.tree.nodes[node]['hops'] >= hops_away_ratio and len(list(self.tree.successors(node))) > 0:
                    self.tree.nodes[node]['type'] = 'boundary'
                    # Make it pretty accurate
                    self.tree.nodes[node]['drift'] = random.uniform(0.000001, 0.00001)  # Very small drift
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
                    if parent:
                        parent = parent[0]

                        # Check if the parent or any ancestor is malicious
                        malicious_ancestor = self.find_malicious_ancestor(node)
                        # Determine which node to synchronize to
                        boundary_clock_ancestor = self.find_boundary_clock_ancestor(node)
                        # See if the boundary clock is closer ancestor than malicious ancestor
                        is_boundary_closer_ancestor = self.is_boundary_clock_below_malicious(malicious_ancestor, boundary_clock_ancestor)

                        # Label the node as attacked
                        if malicious_ancestor and (node not in self.malicious_nodes):
                            self.tree.nodes[node]['has_malicious_ancestor'] = True

                        # Decide if attack impact should come from attacker or boundary clock
                        if self.tree.nodes[node]['has_malicious_ancestor'] and not is_boundary_closer_ancestor:
                            # Apply the malicious impact for the child node (consistent across steps)
                            self.tree.nodes[node]['time'] = self.tree.nodes[node]['susceptibility']
                        elif node not in self.malicious_nodes:
                            # Standard drift
                            self.tree.nodes[node]['time'] += self.tree.nodes[node]['drift'] * sync_interval
                            
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

                            # Simulate delay
                            delay = random.uniform(0.01, 0.05) * hops
                            # Offset calculation
                            offset = self.tree.nodes[node]['time'] - (self.tree.nodes[sync_target]['time'] + delay)
                            
                            # Apply the offset correction
                            self.tree.nodes[node]['time'] -= offset / 2  # Simulate gradual synchronization

            # Call the visualization function if provided
            if visualize_callback:
                visualize_callback(self.tree, self.leader_node, self.boundary_clocks, self.malicious_nodes, step)


    def visualize_sync(self, step, ax, pos, cmap, norm):
        """Visualize the current state of synchronization."""
        ax.clear()

        # Separate nodes by type
        regular_nodes = [node for node in self.tree.nodes if node not in self.boundary_clocks and node not in self.malicious_nodes and not self.tree.nodes[node]['has_malicious_ancestor'] and node != self.leader_node]
        boundary_nodes = [node for node in self.boundary_clocks]
        leader_nodes = [self.leader_node]
        malicious_nodes_list = [node for node in self.malicious_nodes]
        affected_nodes = [node for node in self.tree.nodes if (self.tree.nodes[node]['has_malicious_ancestor'] and (node not in self.malicious_nodes))]

        # Draw edges
        nx.draw_networkx_edges(self.tree, pos, ax=ax, edge_color='grey', node_size=2000)

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

        ax.set_title(f"Step {step}: Clock Synchronization Visualization")
        plt.pause(0.1)

# Main execution
if __name__ == '__main__':
    # # Define the range for the number of nodes and redundancy
    # min_nodes = 100
    # max_nodes = 100
    # min_redundancy = 3

    # # Create the network environment with the specified parameters
    # network_env = NetworkEnvironment(min_nodes, max_nodes, min_redundancy)
    # # Generate and visualize the network
    # graph = network_env.reset()

    # Create a PTP Simulation instance
    clock_sim = ClockSimulation()

    # Create graph
    clock_sim.create_graph()

    # Create minimum spanning tree (MST) and convert it to a directed tree with a root
    clock_sim.reconfigure_tree()

    # Assign initial clock attributes
    clock_sim.assign_initial_clock_attributes()

    # Assign boundary clocks
    clock_sim.assign_boundary_clocks()

    # Select malicious nodes
    clock_sim.select_malicious_nodes(num_malicious=2)

    # Get positions for the original graph
    pos = nx.spring_layout(clock_sim.graph)

    # Set up the plot
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(12, 12))

    # Set up colormap normalization without considering malicious nodes
    valid_times = [clock_sim.tree.nodes[node]['time'] for node in clock_sim.tree.nodes if node not in clock_sim.malicious_nodes]
    norm = mcolors.Normalize(vmin=0, vmax=10)
    cmap = plt.cm.Reds  # Use the 'Reds' colormap for heatmap effect

    # Simulate PTP synchronization and visualize the process live
    clock_sim.simulate_ptp_sync(sync_interval=5, steps=500,
                              visualize_callback=lambda t, l, b, m, s: clock_sim.visualize_sync(s, ax, pos, cmap, norm))

    plt.ioff()  # Disable interactive mode
    plt.show()

    # print("HIIIIIIIIIIIIIIIIIIIIIIIII")

    # # Reconfigure the tree (keeping the same nodes and states)
    # clock_sim.reconfigure_tree()

    # # Simulate PTP synchronization and visualize the process live after reconfiguration
    # clock_sim.simulate_ptp_sync(sync_interval=5, steps=500,
    #                           visualize_callback=lambda t, l, b, m, s: clock_sim.visualize_sync(s, ax, pos, cmap, norm))

    # plt.ioff()  # Disable interactive mode
    # plt.show()



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
