## Savanna Wolvin
# Created: Nov. 8th, 2021
# Edited: 
    
# SUMMARY
# Now let us implement SVM in the dual domain. We use the same dataset, 
# “bank-note.zip”. You can utilize existing constrained optimization libraries.
# For Python, we recommend using “scipy.optimize.minimize”, and you can learn 
# how to use this API from the document at 
# https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.optimize.minimize.html. 
# We recommend using SLSQP to incorporate the equality constraints.

# INPUT 


# OUTPUT



#%% Global Imports
import pandas as pd 
import numpy as np
import scipy.optimize as spo
from datetime import datetime
import itertools as it



#%% Variable Presets

C = [100, 500, 700]
gamma = [0.1, 0.5, 1, 5, 100]

# Data set
data_file_name = 'train_final'
data_file = data_file_name + '.csv'

# column labels
labels = ['age','workclass','fnlwgt','education','educationNum','maritalStatus',
          'occupation','relationship','race','sex','capitalGains',
          'capitalLoss','hoursPerWeek','nativeCountry','incomeGt50']



#%% Load Data

print('Load data and attributes...')
trainData = pd.read_csv(data_file, sep=',', header=0)
trainData = trainData.to_numpy()

testData = pd.read_csv('test_final.csv', sep=',', header=0, index_col=0)
testData = testData.to_numpy()

### pull numeric values
idx_num = []
for idx in range(0, np.shape(trainData)[1]):
    if type(trainData[0,idx]) == int:
        idx_num.append(idx)
trainData = trainData[:, idx_num]
trainData[np.where(trainData[:,6] == 0), 6] = -1

x = trainData[:, range(6)]
y = trainData[:, 6]

idx_num = []
for idx in range(0, np.shape(testData)[1]):
    if type(testData[0,idx]) == int:
        idx_num.append(idx)
testData = testData[:, idx_num]

x2 = testData[:,range(6)]





#%% Functions

def objective_func(a, kernal):    
    cmp_ay = np.reshape(np.multiply(a, y), (-1, 1))
    cmp_ayay = np.matmul(cmp_ay, cmp_ay.T)
    comp_1 = np.multiply(cmp_ayay, kernal)
    comp_1 = (1/2)*np.sum(comp_1)
    
    comp_2 = np.sum(a)
    
    return comp_1 - comp_2


def equality_constraint(a):
    # iteration = [(a[i]*y[i]) for i in range(np.shape(a)[0])]
    iteration = np.multiply(a, y)
    
    return np.sum(iteration)

constraint1 = {'type': 'eq', 'fun': equality_constraint}




#%% Calculate Minimization, Weight Vector, and Bias

for Cx in C:
    if_sv = np.zeros([np.shape(trainData)[0], len(gamma)])
    
    for gammax in gamma:    
        # Bounds
        bnds = [(0, (Cx/873))] * np.shape(trainData)[0]
        
        # Calculate Minimization
        print('Calculate Alpha Values for C = ' + str(Cx) + '/' + str(873) + ' and gamma = ' + str(gammax) + '...')
        
        # Calculate Kernal Array
        print('Calculate Kernel...')
        kernal = np.zeros([np.shape(x)[0], np.shape(x)[0]])
        for i in range(np.shape(x)[0]):
            xi,xj = np.meshgrid(i, range(np.shape(x)[0]))
            xx = np.array(x[xi,:] - x[xj,:], dtype='float')
            kernal[i, :] = np.exp(-(np.linalg.norm(xx, axis=(1,2))**2 / gammax)) 
        
        a0 = [0] * np.shape(x)[0]
        
        try:
            print('Minimize...')
            start_time = datetime.now()
            result = spo.minimize(objective_func, a0, args=(kernal), method='SLSQP', bounds=bnds, constraints=[constraint1], options={'disp': True})
            end_time = datetime.now()
            
            print(result.message + ': ' + 'Duration: ' + str(end_time - start_time))
            print('')
             
            # Calculate Weighted Vector and Bias
            a = result.x
            
            train_prediction = []
            ay = np.multiply(a, y)
            for j in range(np.shape(x)[0]):
                sum_ayk = np.sum(np.multiply(ay, kernal[:,j]))
                train_prediction.append(np.sign(sum_ayk))
                
            train_error = [0]
            [train_error.append(1) for i in range(np.shape(y)[0]) if y[i] != train_prediction[i]]
            train_error = np.sum(train_error) / np.shape(y)[0]
            
            print('Training Error: ' + str(train_error))
            
            
            test_prediction = []
            kernal_test = np.zeros([np.shape(x)[0], np.shape(x2)[0]])
            for i in range(np.shape(x)[0]):
                for j in range(np.shape(x2)[0]):
                    kernalx = np.linalg.norm(x[i,:] - x2[j,:])**2
                    kernalx = -(kernalx / gammax)
                    kernal_test[i, j] = np.exp(kernalx) 
            
            for j in range(np.shape(x2)[0]):
                sum_ayk = np.sum(np.multiply(ay, kernal_test[:,j]))
                test_prediction.append(np.sign(sum_ayk))
            
            numSV = np.sum(a > 0.00001)
            if_sv[a > 0.00001, gamma.index(gammax)] = 1
            print('Number of Support Vectors: ' + str(numSV))
            

            print('')
            print('')
        except: 
            print('Failed for Alpha Values for C = ' + str(Cx) + '/' + str(873) + ' and gamma = ' + str(gammax) + '...')
            print('')
            print('')
        
    comb = it.combinations(range(len(gamma)), 2)
    
    for i in comb:
        same_a = np.sum(if_sv[:, [i[0], i[1]]], axis=1)
        print('For gamma = ' + str(gamma[i[0]]) + ' & ' + str(gamma[i[1]]) + ', there are ' + str(np.sum(same_a == 2)) + ' similar support vectors')
    
    
        
































