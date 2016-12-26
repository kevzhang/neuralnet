import math, copy

def SIGMOID(p):
    def g(a):
        return 1.0 / (1.0 + math.e ** (-a / p))
    return g

class Neuron(object):
    activation = 0
    """List if (neuron, weight) pairs"""
    outbound = []
    sigmoid = None

    def __init__(self, outbound, sig=SIGMOID(1)):
        assert isinstance(outbound, list)
        self.outbound = outbound
        self.sigmoid = sig

    def get_activation(self):
        return self.activation

    def propagate(self):
        self_output = self.sigmoid(self.activation)
        for (neuron, weight) in self.outbound:
            weighted_output = self_output * weight
            neuron.accept_activation(weighted_output)

    def accept_activation(self, activation):
        self.activation += activation

    def reset(self):
        self.activation = 0

    def __str__(self):
        return '[' + str(self.activation) + ' - ' + str(id(self))[-4:] + ']'

class NeuralNet(object):
    """Fully connected for now"""
    input_layer = []
    output_layer = []
    hidden_layers = []
    sigmoid = None

    def __init__(self, num_inputs, num_outputs, hidden_dimensions, sigmoid_p = 1.0):
        self.sigmoid = SIGMOID(sigmoid_p)
        self.output_layer = self.__init_layer(num_outputs, [])
        num_hidden_layers = len(hidden_dimensions)
        hidden_layers = [None] * num_hidden_layers
        for i in range(num_hidden_layers - 1, -1, -1):
            if i == num_hidden_layers - 1:
                outbound_layer = self.output_layer
            else:
                outbound_layer = hidden_layers[i + 1]
            hidden_layers[i] = self.__init_layer(hidden_dimensions[i], outbound_layer) 
        self.hidden_layers = hidden_layers
        self.input_layer = self.__init_layer(num_inputs, hidden_layers[0] if num_hidden_layers else self.output_layer)

    def __init_layer(self, layer_size, outbound_layer):
        layer = [None] * layer_size
        for i in xrange(layer_size):
            # Weights default to 1
            layer[i] = Neuron(zip(outbound_layer, [1] * len(outbound_layer)), sig=self.sigmoid)
        return layer

    def __reset(self):
        for neuron in self.input_layer:
            neuron.reset()
        for layer in self.hidden_layers:
            for neuron in layer:
                neuron.reset()
        for neuron in self.output_layer:
            neuron.reset()

    def __propagate_all_layers(self):
        for neuron in self.input_layer:
            neuron.propagate()
        for layer in self.hidden_layers:
            for neuron in layer:
                neuron.propagate()

    def __set_inputs(self, inputs):
        for i in xrange(len(inputs)):
            neuron = self.input_layer[i]
            activation = inputs[i]
            neuron.accept_activation(activation)

    def intake(self, inputs):
        self.__reset()
        assert len(inputs) == len(self.input_layer),\
            'expecting {0} inputs but got {1}'.format(len(self.input_layer), len(inputs))
        self.__set_inputs(inputs)
        self.__propagate_all_layers()
        output = [neuron.get_activation() for neuron in self.output_layer]
        self.__reset()
        return output

    def __str__(self):
        string = 'input ({0}): '.format(len(self.input_layer))
        for neuron in self.input_layer:
            string += str(neuron)
        string += '\n'
        string += 'hidden with {0} layers: '.format(len(self.hidden_layers))
        for layer in self.hidden_layers:
            string += '\n    '
            for neuron in layer:
                string += str(neuron)
        string += '\n'
        string += 'output ({0}): '.format(len(self.output_layer))
        for neuron in self.output_layer:
            string += str(neuron)
        string += '\n'
        return string