import random

class NodeAttacker:
    
    def __init__(self):
        pass        

    def select_malicious_nodes(self, tree, leader_node, num_malicious=1):
        """
        Select malicious nodes based on the tree structure and leader node.
        Returns a list of malicious node identifiers.
        """
        malicious_nodes = []
        for _ in range(num_malicious):
            # Select nodes that are not the leader and have successors (i.e., non-leaf nodes)
            potential_malicious = [node for node in tree.nodes if node != leader_node and list(tree.successors(node))]
            if potential_malicious:
                # Randomly pick a malicious node from the potential list
                malicious_node = random.choice(potential_malicious)
                malicious_nodes.append(malicious_node)

        return malicious_nodes