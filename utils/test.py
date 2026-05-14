import numpy as np

from sum_tree import SumTree

sum_tree = SumTree(4)

priorities = [2, 4, 5, 3, 500]

for priority in priorities:
    sum_tree.add(priority)

counts = {priority : 0 for priority in priorities}

for i in range(100):
    priority_value = np.random.uniform(0, sum_tree.get_total_priority())
    sampled_node = sum_tree.get_leaf(priority_value)
    counts[sampled_node] += 1