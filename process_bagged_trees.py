## Savanna Wolvin
# Created: Oct. 24th, 2021
# Edited: Oct. 

# SUMMARY

# INPUT

# OUTPUT


#%% Global Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




#%% Variable presets

data_file = 'train_final'
numIterations = 18




#%% import data

trainData = pd.read_csv(data_file + '.csv', sep=',', header=0)
trainData = trainData.to_numpy()
trainOutcome = trainData[:, np.shape(trainData)[1]-1]

bagged = pd.read_csv('census_train_baggedTrees_outcome__attrSubset_2_500.csv', sep=',', 
                                          header=0, index_col=0)
bagged = bagged.to_numpy()
bagged = bagged[:, range(numIterations)]




#%% calculate most likely value

outcome = np.zeros((np.shape(bagged)[0], numIterations))
for itx in range(np.shape(bagged)[1]):
    for row in range(np.shape(bagged)[0]):
        labels, count = np.unique(bagged[row, range(itx+1)], return_counts=1)
        outcome[row, itx] = labels[np.argmax(count)]




#%% calculate error

error = np.zeros((np.shape(outcome)[1]))
for itx in range(np.shape(bagged)[1]):
    error[itx] = sum(abs(outcome[:,itx] - trainOutcome))

error = error / np.shape(bagged)[0]




#%% plot

fig = plt.figure()
plt.plot(range(0,numIterations), error)
plt.xlabel('Number of Bagged Trees')
plt.ylabel('Prediction Error')
plt.title('Precition Error VS Number of Bagged Trees')
plt.legend()
plt.grid()
plt.show()

fig.savefig('error_bagged.png', dpi=300)





