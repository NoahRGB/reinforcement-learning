import numpy as np

class SumTree:
    def __init__(self, size):
        self.size = size
        self.nodes = np.zeros(2 * self.size - 1)
        self.next_data = 0
    
    def add(self, priority):
        node_idx = self.next_data + self.size - 1
        self.update_node(node_idx, priority)
        self.next_data += 1
        if self.next_data >= self.size:
            self.next_data = 0

    def update_node(self, node_idx, priority):
        priority_change = priority - self.nodes[node_idx]
        self.nodes[node_idx] = priority

        # iterate through all parents up the tree and add the new priority change
        current_parent = (node_idx - 1) // 2
        while current_parent >= 0:
            self.nodes[current_parent] += priority_change
            current_parent = (current_parent - 1) // 2

    def get_leaf(self, priority_value):
        current_parent = 0 # start traversal at the root
        left_child = 2 * current_parent + 1
        right_child = left_child + 1
        while left_child < len(self.nodes):

            if priority_value <= self.nodes[left_child]:
                current_parent = left_child
            else:
                priority_value -= self.nodes[left_child]
                current_parent = right_child

            left_child = 2 * current_parent + 1
            right_child = left_child + 1

        return self.nodes[current_parent], current_parent - self.size + 1, current_parent

    def get_total_priority(self):
        return self.nodes[0]