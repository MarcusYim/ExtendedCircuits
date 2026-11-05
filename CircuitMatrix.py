from collections import deque
from typing import Callable, List, Tuple, TypeAlias

import numpy as np
import numpy.typing as npt

Node: TypeAlias = tuple[int, Callable[[float], float]]
EdgeFunc: TypeAlias = Callable[[float], float]
SourceNode: TypeAlias = tuple[int, Callable[[None], float]]
TargetPath: TypeAlias = List[Node]

class CircuitMatrix:
    def __init__(
            self, 
            source_node: SourceNode, 
            modifier_nodes: List[Node], 
            end_nodes: List[Node],
            topology: npt.NDArray[np.int16],
            edge_funcs: List[List[EdgeFunc]],
            ):
        # node funcs map a node input to a node output
        # edge funcs map a node output to a node input
        self.source_node = source_node
        self.modifier_nodes = modifier_nodes
        self.end_nodes = end_nodes
        self.edge_funcs = edge_funcs
        self.topology = topology

        self.end_paths = self.find_connected(topology, source_node, end_nodes)

    def is_valid(self):
        return len(self.end_paths) == len(self.end_nodes)

    def evaluate(self, y0, t):
        pass

    def find_connected(
        A: npt.NDArray[np.int64],
        source: SourceNode,
        targets: list[Node],
    ) -> list[TargetPath]:
        source_num = source[0]
        n = A.shape[0]
        visited = np.zeros(n, dtype=bool)
        parent = np.full(n, -1, dtype=np.int64)

        visited[source_num] = True
        queue = deque([source_num])

        # BFS
        while queue:
            node = queue.popleft()

            for neighbor in np.where(A[node] != 0)[0]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    parent[neighbor] = node
                    queue.append(neighbor)

        # Reconstruct paths only for reachable targets
        results: list[TargetPath] = []

        for target in targets:
            node = target[0]

            if not visited[node]:
                continue

            path = []
            while node != -1:
                path.append(node)
                node = parent[node]

            results.append((target, np.array(path[::-1], dtype=np.int64)))

        return results
