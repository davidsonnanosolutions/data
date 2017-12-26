import mnist_loader
import numpy as np
import network2

training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

net = network2.Network([784, 30, 10], cost=network2.CrossEntropyCost)
net.large_weight_initializer()

test = (net.SGD(training_data, 30, 10, 0.5, evaluation_data=validation_data, monitor_evaluation_accuracy=True))

