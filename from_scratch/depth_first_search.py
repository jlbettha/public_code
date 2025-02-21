""" Depth First Search - uses "Stack" data structure.
    DFS is a tree traversal approach in which the traverse begins at 
    the root node and proceeds through the nodes as far as possible 
    until we reach the node with no unvisited nearby nodes.
    
    DFS builds the tree sub-tree by sub-tree. It works on the 
    concept of LIFO (Last In First Out). DFS is more suitable
    when there are solutions away from source.
    
    DFS is used in various applications such as acyclic graphs and 
    finding strongly connected components etc. There are many 
    applications where both BFS and DFS can be used like Topological 
    Sorting, Cycle Detection, etc.
    
    Time complexity: O(V + E), where V is the number of vertices 
            and E is the number of edges in the graph.
    Space complexity: O(V + E)
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def dfs():
    """_summary_"""


def main() -> None:

    raise NotImplementedError


if __name__ == "__main__":
    tmain = time.time()
    main()
    print(f"Program took {time.time()-tmain:.3f} seconds.")
