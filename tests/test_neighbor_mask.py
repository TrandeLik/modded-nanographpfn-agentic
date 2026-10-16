"""The static-shape isolate fix must preserve graph attention masks exactly."""
import unittest

import torch

from nanographpfn import make_neighbor_mask


class NeighborMaskTests(unittest.TestCase):
    def check_masks(self, device):
        for edges in (torch.empty(2, 0, dtype=torch.long),
                      torch.tensor([[0, 1, 2, 2, 3], [1, 0, 2, 0, 0]])):
            edges = edges.to(device)
            expected = torch.zeros(6, 6, dtype=torch.bool, device=device)
            expected[edges[1], edges[0]] = True
            isolates = ~expected.any(-1)
            indices = torch.where(isolates)[0]
            expected[indices, indices] = True
            actual, actual_isolates = make_neighbor_mask(edges, 6)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual_isolates, isolates, rtol=0, atol=0)

    def test_cpu(self):
        self.check_masks('cpu')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda(self):
        self.check_masks('cuda')
