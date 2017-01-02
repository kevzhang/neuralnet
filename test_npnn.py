import unittest, math
from neural_net.npnn import NeuralNet, SIGMOID

def constant_one():
    return 1

def constant_zero():
    return 0

class TestNeuralNet(unittest.TestCase):

    def test_init_no_error(self):
        NeuralNet(3, 3, [3], sigmoid_p=12.34, sigmoid_s=17.1, weight_provider=constant_one, bias_provider=constant_zero)

    def test_errors_with_bad_input(self):
        nn = NeuralNet(2, 2, [])
        with self.assertRaises(AssertionError):
            nn.intake([[1,2,3]])
        with self.assertRaises(AssertionError):
            nn.intake([1,2])
        with self.assertRaises(AssertionError):
            nn.intake([[[1,2]]])

    def test_single_in_single_out(self):
        nn = NeuralNet(1, 1, [], sigmoid_p=1.0, sigmoid_s=0.0,
                weight_provider=constant_one, bias_provider=constant_zero)
        sigmoid = SIGMOID(1.0, 0)
        self.assertEqual(nn.intake([[1.1]]), [[sigmoid(1.1)]])

    def test_single_in_single_out_bias(self):
        nn = NeuralNet(1, 1, [], sigmoid_p=1.0, sigmoid_s=0.0,
                weight_provider=constant_one, bias_provider=constant_one)
        sigmoid = SIGMOID(1.0, 0)
        self.assertEqual(nn.intake([[1.1]]), [[sigmoid(1.1 + 1) + 1]])

    def test_single_in_single_out_sigmoid(self):
        nn = NeuralNet(1, 1, [], sigmoid_p = 2.0, weight_provider=constant_one, bias_provider=constant_zero)
        sigmoid = SIGMOID(2.0, 0.0)
        self.assertEqual(nn.intake([[1.1]]), [[sigmoid(1.1)]])

    def test_multi_in_multi_out(self):
        nn = NeuralNet(2, 2, [], weight_provider=constant_one, bias_provider=constant_zero)
        sigmoid = SIGMOID(1.0, 0)
        self.assertEqual(nn.intake([[1.1, 1.1]]), [[sigmoid(1.1) * 2] * 2])

    def test_multi_in_multi_out_with_hidden(self):
        nn = NeuralNet(2, 2, [2], weight_provider=constant_one, bias_provider=constant_zero)
        sigmoid = SIGMOID(1.0, 0)
        self.assertEqual(nn.intake([[1.1, 1.1]]), [[sigmoid(sigmoid(1.1) * 2) * 2] * 2])

    def test_error_calculation(self):
        expected = [1,2,3]
        actual = [3,2,1]
        self.assertAlmostEqual(NeuralNet.error(expected, actual), math.sqrt(8), 10)

    def test_squared_error_calculation(self):
        expected_list = [[1,2,3], [2,3,4]]
        actual_list = [[3,2,1], [3,4,5]]
        self.assertAlmostEqual(NeuralNet.squared_error(expected_list, actual_list), 11, 10)

if __name__ == '__main__':
    unittest.main()