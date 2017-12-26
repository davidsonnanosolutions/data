import csv
import pandas as pd
import numpy as np
import magpie_loader as mpl

#mpl.load_data_wrapper()

"""
example = np.arange(10).reshape(10,1)
print example.shape
"""

def testCSV():
	with open('/home/wizard/data/citirineChallenge/training_data.csv', 'rb') as f:
	    test_data = pd.read_csv(f)
	return(test_data)

raw = testCSV()
#test = []
data = [[],[]]
data[0] = raw.ix[:,2:-1]
data[1] = raw.ix[:,-1]

# Code is good to here

'''
i=0
while i < len(data):
	print '{} iteration'.format(i) 
	training_data = [list(data[0][i])]
	i+=1
'''

#training_data = [np.array(data.ix[row,:]) for row in data]
#training_data = list(data.ix[0,:])

i=0
while i < 5:
	print '{} iteration'.format(i) 
	training_data = [np.array(data[0].ix[i,:])]
	training_results = [np.array(data[1].ix[i,:])]
	print '{}'.format(training_results[i])
	i+=1

#print '{}\n{}'.format(training_data[0:5],training_results[0:2])

#training_data = [list(data[0].ix[row,:]) for row in data[0]]	
#print training_data[0:2]

#t1.reshape(t1, (96,1))
#for element in t1:
#	print element
#	
#test.append(raw.ix[:,2:-1])
#test.append(raw.ix[:,-1])

#test_input = [np.reshape(x, (96, 1)) for x in test[0]]




