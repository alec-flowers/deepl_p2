import unittest
from ..module import Linear
import torch


class TestLinear(unittest.TestCase):
    def test_forward(self):
        l = Linear(2, 5, 0, 1)
        x = torch.ones(4, 2)
        out = l.forward(x)
        # matrix sizes are correct
        self.assertEqual(out.size(), (4, 5))
        # matrix multiplication working as expected
        self.assertEqual(out[0, 0], x[0] @ l.w.T[:, 0] + l.b[0])

    def test_backward(self):
        l = Linear(5, 2, 0, 1)
        x = torch.ones(4, 5)
        out = l.forward(x)

        gradwrtoutput = torch.ones(4, 2)
        # Checking sizes for weight update match weight matrix
        self.assertEqual(torch.mm(gradwrtoutput.T, x).size(), l.w.size())

        # Summing correctly applies gradient to bias
        self.assertEqual(torch.sum(gradwrtoutput, dim=0).size(), (2,))

        # gradient to next layer is correct size
        self.assertEqual(torch.mm(gradwrtoutput, l.w).size(), (4, 5))



if __name__ == '__main__':
    unittest.main()
