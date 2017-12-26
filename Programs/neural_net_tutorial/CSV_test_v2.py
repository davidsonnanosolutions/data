import csv
import pandas as pd
import numpy as np
import magpie_loader as mpl

def testCSV():
	with open('/home/wizard/data/citirineChallenge/training_data.csv', 'rb') as f:
	    test_data = pd.read_csv(f)
	return(test_data)

def loadCSV():
	raw = testCSV()
	#test = []
	data = [[],[]]
	data[0] = raw.iloc[:,2:-1] # training input data
	data[1] = raw.iloc[:,-1] # training results
	#print range(0,len(data[1]))
	#print data[1].iloc[0]
	# Code is good to here



	#training_data = [np.array(data[0].ix[i,:]) for i in data[0]]
	training_results = [np.array(data[1].iloc[row]) for row in range(0,len(data[1]))]

	return(training_results)

loadCSV()

print '{}'.format(len(loadCSV()))






