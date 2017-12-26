import numpy as np
import pandas as pd
import random

global training_data_start
global training_data_end
global results_data_position

training_data_start, training_data_end, results_data_position = [2,-1,-1]

with open('/home/wizard/data/citirineChallenge/training_data.csv', 'rb') as f:

        raw_df = pd.read_csv(f)
        #valIndex = np.random.randint(0,len(raw_df),250)
       
        valIndex = random.sample(xrange(len(raw_df)),int(round(0.1*len(raw_df))))
        #print len(raw_df)

        validationSample = pd.DataFrame(raw_df.ix[i,2:-1] for i in valIndex).sort_index(ascending=True)

        input_df = raw_df.drop(valIndex)
        input_df.reset_index()
        print input_df
        '''for j in range(0,len(validationSample):
        	raw_df.drop(j, inplace=True)'''

        #input_df = raw_df[raw_df.isin(validationSample) == False]
        #print len(raw_df)
        #validationSample = validationSample.ix[]
        #print validationSample

        inputData = []
        inputData.append(raw_df.iloc[:,training_data_start:training_data_end])
        inputData.append(raw_df.iloc[:,results_data_position])
