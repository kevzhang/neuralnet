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

def train_until(neural_net, training_data, initial_step=0.1, threshold=0.1):
    history = RollingErrorHistory(10)

    current_step = initial_step
    prev_squared_error = neural_net.get_training_data_squared_error(training_data)
    print 'initial_squared_error', prev_squared_error
    history.record_and_compare(prev_squared_error)
    while True:
        squared_error = neural_net.train(training_data, step=current_step)
        progress = history.record_and_compare(squared_error)
        if (progress != None and abs(progress) < threshold):
            print 'progress in last 10 rounds = ' + str(progress) + ' - halt training'
            break
        print 'squared_error', squared_error, 'step', current_step
        if squared_error < prev_squared_error:
            current_step *= 1.2
        else:
            current_step /= 4
        prev_squared_error = squared_error


