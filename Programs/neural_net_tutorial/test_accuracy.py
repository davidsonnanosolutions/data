import numpy as np
import random

a = np.arange(10)
b = np.arange(10)
random.shuffle(b)


results = zip(a,b)
print results

#results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
accuracy =  np.sum(np.int(x == y) for (x, y) in results)
print accuracy

(0, array([[ 1.],
       [ 0.],
       [ 0.],
       [ 1.],
       [ 0.],
       [ 0.],
       [ 0.],
       [ 0.],
       [ 0.],
       [ 0.],
       [ 1.]]))