## Test Sigmoid ##

import Sigmoid as sig
import numpy as np

k=0
x=[]

while k <= 1:
	x.append(k)
	k+=0.1

j=[1,2,3,4,5,6,7,8,9,1000]

## Loop through and evaluate z at each value, store in an array and print.

sub = [sig.sigmoid(h) for h in x]
ove = [sig.sigmoid(h) for h in j]

a = zip(x,sub)
b = zip(j,ove)

print 'Between 0 and 1:'
for row in a: print row

print 'Between 1 and 3:'
for row in b: print row
