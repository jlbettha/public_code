"""Breadth First Search - uses "Queue" data structure for
finding the shortest path. BFS is a traversal approach
in which we first walk through all nodes on the same
level before moving on to the next level.

BFS builds the tree level by level. It works on the
concept of FIFO (First In First Out). BFS is more
suitable for searching vertices closer to the given source.

BFS is used in various applications such as bipartite graphs,
shortest paths, etc. If weight of every edge is same, then
BFS gives shortest pat from source to every other vertex.

Time Complexity: O(V+E), where V is the number of nodes
        and E is the number of edges.
Space Complexity: O(V)
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray


def breadth_first_search() -> None:

    raise NotImplementedError


def main() -> None:

    raise NotImplementedError


if __name__ == "__main__":
    tmain = time.time()
    main()
    print(f"Program took {time.time()-tmain:.3f} seconds.")
