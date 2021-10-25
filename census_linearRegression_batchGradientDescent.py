## Savanna Wolvin
# Created: Oct. 5th, 2021
# Edited: Oct. 20th, 2021

# SUMMARY
# Implement the batch gradient descent algorithm, and tune the learning rate r
# to ensure the algorithm converges.  To examine convergence, you can watch 
# the norm of the weight vector difference,‖wt−wt−1‖, at each stept.  if 
# ‖wt−wt−1‖is less than a tolerance level, say, 10−6, you can conclude that it 
# converges.  You can initialize your weight vector to be 0.  Please find an 
# appropriate r such that the  algorithm  converges. To  tune r,  you  can  
# start  with a relatively big value, say, r= 1, and then gradually decrease 
# r, say r= 0.5,0.25,0.125,..., until you see the convergence. Report the 
# learned weight vector, and the learning rate r. Meanwhile, please record the
# cost function value of the training data at each step, and then draw a 
# figure shows how the cost function changes along with steps. Use your final 
# weight vector to calculate the cost function value of the test data.

# INPUT
# data_file - file location that contains the training data to create the 
#                   decision tree
# labels    - list of column labels used by the data_file
# r         - Learning Rate
# T         - Number of Iterations

# OUTPUT
# 'cost_vs_iterations_4a.png' - PNG File Showing the Relationship Between the 
#           Cost Function Value and the Iteration Number for the Training Data




#%% Global Imports
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt




#%% Variable Presets

# Data set
data_file_name = 'train_final'
data_file = data_file_name + '.csv'

# column labels
labels = ['age','workclass','fnlwgt','education','educationNum','maritalStatus',
          'occupation','relationship','race','sex','capitalGains',
          'capitalLoss','hoursPerWeek','nativeCountry','incomeGt50']

# learning rate
r = 0.000089

# number of iterations
T = 500




#%% Empty Variables

w_matrix = np.zeros([6, T+1])
norm        = np.zeros([T])
cost_func   = np.zeros([T])




#%% Main

def main():
    ### Load Data
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
    
    idx_num = []
    for idx in range(0, np.shape(testData)[1]):
        if type(testData[0,idx]) == int:
            idx_num.append(idx)
    testData = testData[:, idx_num]
    
    ### Normalize data
    trainData = (trainData - np.min(trainData, axis=0)) / (np.max(trainData, axis=0) - np.min(trainData, axis=0))
    testData = (testData - np.min(testData, axis=0)) / (np.max(testData, axis=0) - np.min(testData, axis=0))
    
    data_lngth = np.shape(trainData)[1]-1
    
    batchGradient = np.zeros([T, np.shape(w_matrix)[0]])
    
    for tx in range(T):  # each iteration
        ### Calculate gradient for each component in the W matix
        for wi in range(np.shape(w_matrix)[0]):  # each w value
            xx = 0
            
            for ex in range(np.shape(trainData)[0]): # each example
                xx += (trainData[ex,data_lngth] - np.dot(w_matrix[:,tx], trainData[ex,range(0,data_lngth)])) * (trainData[ex,wi])
        
            batchGradient[tx, wi] = -xx
            
        
        ### Update W Matrix
        for wi in range(np.shape(w_matrix)[0]):
            w_matrix[wi, tx+1] = w_matrix[wi, tx] - (r * batchGradient[tx, wi])
            
        ### Report Norm of Weight Vector Difference
        diff_w      = w_matrix[:, tx+1] - w_matrix[:, tx]
        norm[tx]    = np.linalg.norm(diff_w)
        # print('Norm of Weight Vector: ' + str(round(norm[tx], 6)))
        
        ### Report the Cost Function
        cost_func[tx] = costFunction(trainData, w_matrix, tx+1)
        # print('Cost Function Value: ' + str(round(cost_func[tx], 6)))
        
        
    # test_cost_function = testCostFunction(w_matrix, tx+1)
    ### Predictions
    prdctn_train    = np.zeros([np.shape(trainData)[0]])
    prdctn_test     = np.zeros([np.shape(testData)[0]])
    
    for idx in range(np.shape(trainData)[0]):
        prdctn_train[idx] = round(np.dot(w_matrix[:,tx+1], trainData[idx,range(6)]))
    
    for idx in range(np.shape(testData)[0]):
        prdctn_test[idx] = round(np.dot(w_matrix[:,tx+1], testData[idx,:]))
    
    error = sum(abs(prdctn_train - trainData[idx,6])) / np.shape(trainData)[0]
        
    ### Plot Cost Function
    plt.plot(range(T), cost_func)
    plt.title('Cost Function VS Iterations for Training Data, rate: ' + str(r))
    plt.ylabel('Cost Function')
    plt.xlabel('Iterations')
    plt.grid()
    plt.show
    # plt.savefig('cost_vs_iterations_4a.png', dpi = 300)
    
    print('Final Cost Value: ' + str(cost_func[tx]))
    # print('Test Cost Value: ' + str(test_cost_function))
    print('Final Norm Value: ' + str(norm[tx]))
    print(w_matrix[:,tx+1])
    print(error)




#%% Calculate Value of Cost Function with New Values

def costFunction(trainData, w_matrix, tx):
    value = 0
    data_lngth = np.shape(trainData)[1]-1
    
    for ex in range(np.shape(trainData)[0]): # each example
        value += ( trainData[ex, data_lngth] - np.dot(w_matrix[:,tx], trainData[ex,range(0,data_lngth)]) )**2

    value = value * (0.5)
    return value




#%% Calculate the Cost Function of the Test Data

def testCostFunction(w_matrix, tx):
    testData = pd.read_csv('concrete/test.csv', sep=',', header=None)
    testData = testData.to_numpy()
    data_lngth = np.shape(testData)[1]-1
    
    value = 0
    data_lngth = np.shape(testData)[1]-1
    
    for ex in range(np.shape(testData)[0]): # each example
        value += ( testData[ex, data_lngth] - np.dot(w_matrix[:,tx], testData[ex,range(0,data_lngth)]) )**2
    
    value = value * (0.5)
    
    return value




#%% Main
main()



