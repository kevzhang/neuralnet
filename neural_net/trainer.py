from Queue import Queue

class RollingErrorHistory(object):
    history = []
    history_size = 0

    def __init__(self, history_size):
        self.history_size = history_size

    def __record_error(self, error):
        if (len(self.history) < self.history_size):
            self.history.append(error)
        else:
            self.history = self.history[1:]
            self.history.append(error)

    def record_and_compare(self, error):
        self.__record_error(error)
        
        if (len(self.history) == self.history_size):
            return self.history[0] - self.history[-1]
        else:
            return None

# Threshold is still with respect to total squared error
def train_until(neural_net, training_data, initial_step=0.1, threshold=0.1):
    history = RollingErrorHistory(6)
    num_examples = len(training_data)

    current_weight_step = initial_step
    current_bias_step = initial_step

    prev_squared_error = neural_net.get_training_data_squared_error(training_data)
    history.record_and_compare(prev_squared_error)
    print 'avg_sq_error', prev_squared_error / num_examples

    while True:
        # Train edge weights
        weight_squared_error = neural_net.train_weight(training_data, step=current_weight_step)
        if weight_squared_error < prev_squared_error:
            current_weight_step *= 1.2
        else:
            current_weight_step /= 4
        print 'after_weights', weight_squared_error / num_examples
        # Train neuron bias
        bias_squared_error = neural_net.train_bias(training_data, step=current_bias_step)
        if bias_squared_error < weight_squared_error:
            current_bias_step *= 1.2
        else:
            current_bias_step /= 4
        print 'after_bias', bias_squared_error / num_examples

        final_squared_error = bias_squared_error
        progress = history.record_and_compare(final_squared_error)
        if (progress != None and abs(progress) < threshold):
            print 'progress in last 10 rounds = ' + str(progress) + ' - halt training'
            break
        print 'step_sizes: edge-step', current_weight_step, 'bias-step', current_bias_step
        prev_squared_error = final_squared_error


