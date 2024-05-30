import networkx as nx
import numpy as np


class LaserDynamicTransform:

    additions_factor: float = 1
    minimum_additions: int = 1
    shuffle: bool = True

    def __init__(self, G, num_snapshots, edges):
        self.G = G
        self.num_snapshots = num_snapshots
        self.edges = edges

    def create_rewirings(self):
        A = nx.adjacency_matrix(self.G).toarray() + np.eye(self.G.number_of_nodes())
        A[A == 2] = 1 # in case the matrix already had a diagonal
        D = A
        A_curr = A 
        A_next = A @ A

        for r in range(2, self.num_snapshots + 1):
            # clip matrices to 1 and 0
            A_next[A_next != 0] = 1 
            A_curr[A_curr != 0] = 1

            A_difference = A_next - A_curr
            D_difference = r * A_difference
            D += D_difference
            
            A_curr = A_next
            A_next = A_next @ A
    
        D = np.tril(D)
        M = np.linalg.matrix_power(A, 8)  # ((A^2)^2)^2
        return [self._get_new_edges(D, M, r) for r in range(2, self.num_snapshots + 1)]
    
    def _get_new_edges(self, D, M, r):
        """
        D: Distance matrix (up to self.num_snapshots)
        M: Connectivity matrix (in this case, power of adjacency matrix)
        r: Current snapshot
        """

        n = D.shape[0]
        added_edges = []
        for i in range(n):
            D_row, M_row = D[i,:], M[i,:]
            D_row_mask = D_row == r
            D_row_mask[i] = False

            M_row_masked = M_row[D_row_mask]
            orbit_size = M_row_masked.shape[0]
            add_per_node = max(self.minimum_additions, round(orbit_size * self.additions_factor))

            best_k_idxs = self._get_best_k_idxs(M_row_masked, add_per_node)

            D_mask_indices = np.where(D_row_mask)
            selected = np.array(D_mask_indices)[0][best_k_idxs]

            for j in selected:
                added_edges.append([i, j])
        return np.array(added_edges)
    

    def _get_best_k_idxs(self, M_masked, k):
        """
        Sample k indices given the connectivity measure. We sample
        from the first k_idxs. If `self.shuffle`, then we add very 
        small noise in such a way to break tie-breaks uniformly.

        Args:
            M_masked: The connectivity measures.
            k: Number of indices to sample.
        """
        size = M_masked.shape[0]
        
        # To break tie breaks, add very small IID noise
        # such that it randomly shuffles only elements with the 
        # same value (allows us to efficiently do random sampling)
        if self.shuffle:
            eps = 10e-8
            M_masked = M_masked + np.random.normal(0, eps, size)

        if size <= k:
            lowest_k_idxs = np.arange(size)
        else:
            lowest_k_idxs = np.argpartition(M_masked, k)[:k]

        return lowest_k_idxs