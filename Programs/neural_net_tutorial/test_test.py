import random

# Third-party libraries
import numpy as np
import pandas as pd

global training_data_start
global training_data_end
global results_data_position

training_data_start, training_data_end, results_data_position = [2,-1,-1]

def testLoader():

    with open('/home/wizard/data/citirineChallenge/training_data.csv', 'rb') as f:

        raw_df = pd.read_csv(f)
        
        valIndex = random.sample(xrange(len(raw_df)),int(round(0.1*len(raw_df))))
        validation_df = pd.DataFrame(raw_df.ix[i,2:-1] for i in valIndex).sort_index(ascending=True)
        validation_df.reset_index()

        input_df = raw_df.drop(valIndex)
        input_df.reset_index()

        input_data = []
        input_data.append(input_df.iloc[:,training_data_start:training_data_end])
        input_data.append(input_df.iloc[:,results_data_position])

        validation_data = []
        validation_data.append(validation_df.iloc[:,training_data_start:training_data_end])
        validation_data.append(validation_df.iloc[:,results_data_position])

    with open('/home/wizard/data/citirineChallenge/test_data.csv', 'rb') as f:

        test_df = pd.read_csv(f)

        test_data = []
        test_data.append(test_df.iloc[:,training_data_start:training_data_end])
        test_data.append(test_df.iloc[:,results_data_position])

    return(input_data, validation_data, test_data)

ind,vs,td = testLoader()

print 'input: {}, validation: {}, test: {}'.format(len(ind[0]),len(vs[0]),len(td[0]))