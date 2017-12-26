import numpy as np

#a = np.zeros((10,3))
#print len(a[0])
a = np.array([0,0,0])
a = list(a)
print a

results = [(a,a),(1,1),(2,2)]
print np.sum(np.int(x == y) for (x, y) in results)
