import unittest
from nn import Neuron, NeuralNet, SIGMOID

class TestNeuralNet(unittest.TestCase):

    def test_init_no_error(self):
        NeuralNet(3, 3, [3])

    def test_errors_with_bad_input(self):
        nn = NeuralNet(2, 2, [])
        with self.assertRaises(AssertionError):
            nn.intake([1,2,3])

    def test_single_in_single_out(self):
        nn = NeuralNet(1, 1, [])
        sigmoid = SIGMOID(1.0)
        self.assertEqual(nn.intake([1.1]), [sigmoid(1.1)])

    def test_single_in_single_out_sigmoid(self):
        nn = NeuralNet(1, 1, [], sigmoid_p = 2.0)
        sigmoid = SIGMOID(2.0)
        self.assertEqual(nn.intake([1.1]), [sigmoid(1.1)])

    def test_multi_in_multi_out(self):
        nn = NeuralNet(2, 2, [])
        sigmoid = SIGMOID(1.0)
        self.assertEqual(nn.intake([1.1, 1.1]), [sigmoid(1.1) * 2] * 2)

    def test_multi_in_multi_out_with_hidden(self):
        nn = NeuralNet(2, 2, [2])
        sigmoid = SIGMOID(1.0)
        self.assertEqual(nn.intake([1.1, 1.1]), [sigmoid(sigmoid(1.1) * 2) * 2] * 2 )

if __name__ == '__main__':
    unittest.main()