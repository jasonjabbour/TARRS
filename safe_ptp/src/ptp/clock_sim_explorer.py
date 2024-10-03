import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import random

from safe_ptp.src.ptp.clock_sim import ClockSimulation
from safe_ptp.src.env.node_attacker import NodeAttacker

def test_reconfiguring_tree():
    # Create a PTP Simulation instance
    clock_sim = ClockSimulation(render=True, seed=40)

    # Create an attacker
    node_attacker = NodeAttacker()
    
    # Select and set malicious nodes using the attacker class
    malicious_nodes = node_attacker.select_malicious_nodes(clock_sim.tree, clock_sim.leader_node, num_malicious=2)

    # Set the attacked nodes attributes    
    clock_sim.set_malicious_attributes(malicious_nodes)

    # Simulate PTP synchronization and visualize the process live
    clock_sim.simulate_and_render(sync_interval=5, steps=50)

    # You can also reconfigure the graph
    num_reconfigurations = 10
    for _ in range(num_reconfigurations):
        # Reconfigure the tree (keeping the same nodes and states)
        clock_sim.reconfigure_tree(randomize_weights=True)

        # Select node to disconnect 
        clock_sim.randomly_disconnect_nodes()

        # Simulate PTP synchronization and visualize the process live after reconfiguration
        clock_sim.simulate_and_render(sync_interval=5, steps=50)

    clock_sim.finalize_render()

def test_reconfiguring_tree_from_action():
    # Create a PTP Simulation instance
    clock_sim = ClockSimulation(community_size=7, community_num=7, render=True, seed=None)

    # Create an attacker
    node_attacker = NodeAttacker()
    
    # Select and set malicious nodes using the attacker class
    malicious_nodes = node_attacker.select_malicious_nodes(clock_sim.tree, clock_sim.leader_node, num_malicious=2)
    
    # Set the attacked nodes attributes    
    clock_sim.set_malicious_attributes(malicious_nodes)
    
    # Simulate PTP synchronization and visualize the process live
    clock_sim.simulate_and_render(sync_interval=5, steps=20)

    print("Total Desync Time", clock_sim.get_total_desync_time())
    # print(clock_sim.get_state_features())
    # print(clock_sim.get_tree_edge_indices())

    # You can also reconfigure the graph
    num_reconfigurations = 10
    for _ in range(num_reconfigurations):
        # Get the number of undirected edges in the physical network graph
        num_physical_edges = clock_sim.graph.number_of_edges()

        # Generate a mock action vector representing which of the edges in the physical network will be in the tree
        # For testing purposes, this vector will randomly decide if each edge is included (0 or 1)
        action_vector = [1 if random.random() > 0.5 else 0 for _ in range(2*num_physical_edges)]  # Random binary decisions

        # Reconfigure the tree using the action vector
        clock_sim.construct_tree_from_edge_vector(action_vector)

        # Simulate PTP synchronization and visualize the process live after reconfiguration
        clock_sim.simulate_and_render(sync_interval=5, steps=20)

        print("Total Desync Time", clock_sim.get_total_desync_time())
        # print(clock_sim.get_state_features())
        # print(clock_sim.get_tree_edge_indices())

    clock_sim.finalize_render()

# Main execution
if __name__ == '__main__':
    # Test a reconfigured tree after N simulation steps
    # test_reconfiguring_tree()

    # Test a specific tree constructed using an action vector 
    test_reconfiguring_tree_from_action()
    