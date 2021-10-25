## Savanna Wolvin
# Created: Oct. 24th, 2021
# Edited: Oct. 

# SUMMARY

# INPUT

# OUTPUT


#%%  Global Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




#%% Variable presets

data_file = 'train_final'
iterationMax = 14




#%% import data

trainData = pd.read_csv(data_file + '.csv', sep=',', header=0)
trainData = trainData.to_numpy()
trainOutcome = trainData[:, np.shape(trainData)[1]-1]

unknownNotAttr = np.zeros((np.shape(trainData)[0], iterationMax-1))
unknownAsAttr = np.zeros((np.shape(trainData)[0], iterationMax-1))

for idx in range(2,iterationMax+1):
    xx = pd.read_csv('cencus_dt_train_Entropy_' + str(idx) + '_unknownNotAttr.csv', sep=',', 
                                          header=0, index_col=0)
    xx = xx.to_numpy()
    unknownNotAttr[:,idx-2] = xx[:,0]
    
    xx = pd.read_csv('cencus_dt_train_Entropy_' + str(idx) + '_unknownAsAttr.csv', sep=',', 
                                         header=0, index_col=0)
    xx = xx.to_numpy()
    unknownAsAttr[:,idx-2] = xx[:,0]
    



#%% Calculate error

error_unknownNotAttr = np.zeros((np.shape(unknownNotAttr)[1]))
error_unknownAsAttr  = np.zeros((np.shape(unknownNotAttr)[1]))
for idx in range(np.shape(unknownNotAttr)[1]):
    error_unknownNotAttr[idx] = sum(abs(unknownNotAttr[:,idx] - trainOutcome))
    error_unknownAsAttr[idx]  = sum(abs(unknownAsAttr[:,idx]  - trainOutcome))

error_unknownNotAttr = error_unknownNotAttr / np.shape(unknownNotAttr)[0]
error_unknownAsAttr  = error_unknownAsAttr  / np.shape(unknownNotAttr)[0]




#%% plot

fig = plt.figure()
plt.plot(range(2,iterationMax+1), error_unknownAsAttr, label='Unknown as an Attribute')
plt.plot(range(2,iterationMax+1), error_unknownNotAttr, label='Unknown not an Attribute')
plt.xlabel('Decision Tree Level')
plt.ylabel('Prediction Error')
plt.title('Precition Error VS Tree Level')
plt.legend()
plt.grid()
plt.xticks([2,3,4,5,6,7,8,9,10,11,12,13,14])
plt.show()

fig.savefig('error_unknownAttr.png', dpi=300)



#%% error difference

fig2 = plt.figure()
plt.plot(range(2,iterationMax+1), (error_unknownNotAttr - error_unknownAsAttr))
plt.xlabel('Decision Tree Level')
plt.ylabel('Prediction Error Difference')
plt.title('Unknown Not an Attribute - Unknown as an Attribute')
plt.legend()
plt.grid()
plt.xticks([2,3,4,5,6,7,8,9,10,11,12,13,14])
plt.show()

fig2.savefig('error_diff.png', dpi=300)















































