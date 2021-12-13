## Savanna Wolvin
# Created: Oct. 4th, 2021
# Edited: Sep. 10th, 2021

# SUMMARY
# Implement the random forest algorithm as we discussed in our lecture. Vary 
# the number of random trees from 1 to 500. Note that you need to modify your 
# tree learning algorithm to randomly select a subset of features before each 
# split. Then use the information gain to select the best feature to split. 
# Vary the size of the feature subset from {2,4,6}. Report in a figure how the 
# training and test errors vary along with the number of random trees for each 
# feature subsetsize setting. How does the performance compare with bagged 
# trees?

# INPUT
# maxTreeDepth  - maximum number of levels on the decision tree
# algorithmType - choose between three algorithm types to create the decision 
#                   tree
# data_file     - file location that contains the training data to create the 
#                   decision tree
# labels        - list of column labels used by the data_file
# isCategory    - indicate if you want to use 'unknown' as its own attribute 
#                   feature or not

# OUTPUT
# 'car_decision_tree.csv' - CSV File Containing the Attributes, Catagories, 
#                           and Outcomes of the Decision Tree




#%% Global Imports
import pandas as pd 
import numpy as np
import sys
import random as rd
# import matplotlib.pyplot as plt
# import itertools as it




#%% Variable Presets

# 1 through 16
maxTreeDepth = 16

# 'Entropy', 'GiniIndex', 'MajorityError'
algorithmType = 'Entropy'

# number of iterations
T = 500

# Uniformly WITH Replacement
replacement = True
n_samples = 5000

# Number of Feature Subset
num_subset = 2

# Data set
data_file_name = 'train_final'
data_file = data_file_name + '.csv'

# column labels
labels = ['age','workclass','fnlwgt','education','educationNum','maritalStatus',
          'occupation','relationship','race','sex','capitalGains',
          'capitalLoss','hoursPerWeek','nativeCountry','incomeGt50']
# labels = ['Outlook','Temperature','Humidity','Winds','Play?']
    
# Use Unknown As A Particular Attribute Value
isCategory = True




#%% Main Function

def main():
    ### Load Data
    print('Load data and attributes...')
    trainData = pd.read_csv(data_file, sep=',', header=0)
    trainData = trainData.to_numpy()
    
    testData = pd.read_csv('test_final.csv', sep=',', header=0, index_col=0)
    testData = testData.to_numpy()
    
    ### Use 'Unknown' As A Particular Attribute Value
    if not(isCategory):
        trainData = replaceUnknowns(trainData)
        testData = replaceUnknowns(testData)
    
    ### Create Dictionary and Change Numeric Values Into Categorical Values
    attr_dict = {}
    for idx in range(0, np.shape(trainData)[1]-1):
        if type(trainData[0,idx]) == int:
            if idx == 10 or idx == 11:
                attr_dict.update({labels[idx]: ['none','q1','q2','q3','q4']})
                
                num_loc = np.where(trainData[:,idx] > 0)
                non_loc = np.where(trainData[:,idx] == 0)
                quartiles  = np.quantile(trainData[num_loc,idx], [0.25, 0.5, 0.75])
                q1 = np.where(trainData[:,idx] < quartiles[0])
                q2 = np.where((trainData[:,idx] < quartiles[1]) & (trainData[:,idx] >= quartiles[0]))
                q3 = np.where((trainData[:,idx] < quartiles[2]) & (trainData[:,idx] >= quartiles[1]))
                q4 = np.where(trainData[:,idx] >= quartiles[2])
                trainData[q1,idx] = 'q1'
                trainData[q2,idx] = 'q2'
                trainData[q3,idx] = 'q3'
                trainData[q4,idx] = 'q4'
                trainData[non_loc,idx] = 'none'
                
                num_loc = np.where(testData[:,idx] > 0)
                non_loc = np.where(testData[:,idx] == 0)
                quartiles  = np.quantile(testData[num_loc,idx], [0.25, 0.5, 0.75])
                q1 = np.where(testData[:,idx] < quartiles[0])
                q2 = np.where((testData[:,idx] < quartiles[1]) & (testData[:,idx] >= quartiles[0]))
                q3 = np.where((testData[:,idx] < quartiles[2]) & (testData[:,idx] >= quartiles[1]))
                q4 = np.where(testData[:,idx] >= quartiles[2])
                testData[q1,idx] = 'q1'
                testData[q2,idx] = 'q2'
                testData[q3,idx] = 'q3'
                testData[q4,idx] = 'q4'
                testData[non_loc,idx] = 'none'
                
            else:
                attr_dict.update({labels[idx]: ['q1','q2','q3','q4']})  
                
                quartiles  = np.quantile(trainData[:,idx], [0.25, 0.5, 0.75])
                q1 = np.where(trainData[:,idx] < quartiles[0])
                q2 = np.where((trainData[:,idx] < quartiles[1]) & (trainData[:,idx] >= quartiles[0]))
                q3 = np.where((trainData[:,idx] < quartiles[2]) & (trainData[:,idx] >= quartiles[1]))
                q4 = np.where(trainData[:,idx] >= quartiles[2])
                trainData[q1,idx] = 'q1'
                trainData[q2,idx] = 'q2'
                trainData[q3,idx] = 'q3'
                trainData[q4,idx] = 'q4'
            
                quartiles  = np.quantile(testData[:,idx], [0.25, 0.5, 0.75])
                q1 = np.where(testData[:,idx] < quartiles[0])
                q2 = np.where((testData[:,idx] < quartiles[1]) & (testData[:,idx] >= quartiles[0]))
                q3 = np.where((testData[:,idx] < quartiles[2]) & (testData[:,idx] >= quartiles[1]))
                q4 = np.where(testData[:,idx] >= quartiles[2])
                testData[q1,idx] = 'q1'
                testData[q2,idx] = 'q2'
                testData[q3,idx] = 'q3'
                testData[q4,idx] = 'q4'
        else:
            attr_dict.update({labels[idx]: np.unique(trainData[:,idx]).tolist()})
    
    ### Array to Hold Outcomes
    dtOutcome_all_train = np.zeros([np.shape(trainData)[0],T], dtype=object)
    dtOutcome_all_test = np.zeros([np.shape(testData)[0],T], dtype=object)
    
    ### Loop Through Each Iteration to Create a Forest of Stumps
    for tx in range(0, T):
        print('Iteration ' + str(tx+1))
        
        ### Create Dataset
        trainDataBagged = drawDataSamples(trainData, replacement, n_samples)
        
    
        ### Determine Head Node & Create Data Frame Containing Decision Tree
        # print('Determine Head Node...')
        rand_avail_attributes    = rd.sample(range(0, len(labels)-1), num_subset)         # pick random attributes
        headNode            = pickAttribute(trainDataBagged, rand_avail_attributes )
        decisionTree_attr   = np.array([labels[headNode]] * len(attr_dict[labels[headNode]]), ndmin=2)
        decisionTree_ctgr   = np.array(attr_dict[labels[headNode]], ndmin=2)
        
    
        ### Loop to Create a Greater Than One Level Decision Tree
        level = 2
        while np.shape(decisionTree_attr)[0] < (maxTreeDepth) and np.shape(decisionTree_attr)[0] < (len(labels)-1):
            # print('Determine ' + str((np.shape(decisionTree_attr)[0])+1) + ' Layer...')
            data_lngth = np.shape(trainDataBagged)[0]
            
            ### Create Temporary Arrays
            decisionTree_attrX = np.zeros((np.shape(decisionTree_attr)[0]+1,0))
            decisionTree_ctgrX = np.zeros((np.shape(decisionTree_ctgr)[0]+1,0))
            
            ### Loop Through Each Available Attribute Combination ###
            for branchX in range(0, np.shape(decisionTree_attr)[1]):
                ### Determine Used and Available Attributes
                used_attributes, avail_attributes = whichAttributes(decisionTree_attr, branchX)
                
                if len(list(avail_attributes)) > num_subset:
                    rand_avail_attributes    = rd.sample(list(avail_attributes), num_subset)
                else:
                    rand_avail_attributes = avail_attributes
                    
                ### Determine if Another Row Is Needed
                if needAnotherNode(trainDataBagged, used_attributes, decisionTree_ctgr[:,branchX]):
                    ### Determine Next Node
                    decision_branch_idx = [i for i in range(data_lngth) if 
                                      np.array_equal(trainDataBagged[i, used_attributes], decisionTree_ctgr[:,branchX])]
                    trainDataX  = trainDataBagged[:, np.append(rand_avail_attributes,(len(labels)-1)).tolist()]
                    branch_attr = pickAttribute(trainDataX[decision_branch_idx,:], rand_avail_attributes)
                    
                    ### Add Attribute to Branch
                    xx                  = np.column_stack(
                        [[decisionTree_attr[:,branchX]] * len(attr_dict[labels[branch_attr]]), 
                         [labels[branch_attr]]* len(attr_dict[labels[branch_attr]])])
                    decisionTree_attrX  = np.column_stack([decisionTree_attrX, xx.T])
                    
                    xx                  = np.column_stack(
                        [[decisionTree_ctgr[:,branchX]] * len(attr_dict[labels[branch_attr]]),
                         np.array(attr_dict[labels[branch_attr]], ndmin=2).T])
                    decisionTree_ctgrX  = np.column_stack([decisionTree_ctgrX, xx.T])
                else:
                    # print('End of Branch')
                    xx = np.column_stack([[decisionTree_attr[:,branchX]], ['']])
                    decisionTree_attrX = np.column_stack([decisionTree_attrX, xx.T])
                    
                    xx = np.column_stack([[decisionTree_ctgr[:,branchX]],['']])
                    decisionTree_ctgrX = np.column_stack([decisionTree_ctgrX, xx.T])
                
            ### Move Temporary Arrays into Permanent Arrays
            decisionTree_attr = decisionTree_attrX
            decisionTree_ctgr = decisionTree_ctgrX
        
            level += 1
        
        
        ### Find Decision Tree Outcome
        dtOutcome_train, dtOutcome_test = mostLikelyOutcome(decisionTree_attr, decisionTree_ctgr, trainData, testData)
        dtOutcome_all_train[:,tx] = dtOutcome_train[:,0]
        dtOutcome_train[:] = ''
                
        
        dtOutcome_all_test[:,tx] = dtOutcome_test[:,0]
        
        
        pd.concat([pd.DataFrame(dtOutcome_all_train)]).to_csv(
                'census_train_baggedTrees_outcome_' + 
                '_attrSubset_' + str(num_subset) + '_500.csv', index = True, header = True)
        
        pd.concat([pd.DataFrame(dtOutcome_all_test)]).to_csv(
                'census_test_baggedTrees_outcome_' + 
                '_attrSubset_' + str(num_subset) + '_500.csv', index = True, header = True)
            
    
    
    

#%% Replace the Unknown Value with the Most Common Value

def replaceUnknowns(trainData):
    for attrX in range(0, np.shape(trainData)[1]):
        attr_ctgrs, attr_cnt = np.unique(trainData[:,attrX], return_counts=1)
        mostUsed = attr_ctgrs[int(np.argmax(attr_cnt))]
        
        for idx in range(0, np.shape(trainData)[0]):
            if trainData[idx, attrX] == '?':
                trainData[idx, attrX] = mostUsed

    return trainData




#%% Bagged Dataset

def drawDataSamples(trainData, replacement, n_samples):
    trainDataX  = np.zeros((1,0))
    data_lngth  = np.shape(trainData)[0]
    
    if replacement:
        xx = [rd.randint(0, data_lngth-1) for i in range(0, n_samples)]
        trainDataX = trainData[xx, :]
            
    else:
        xx = rd.sample(range(0, data_lngth-1), n_samples)   
        trainDataX = trainData[xx, :]
    
    return trainDataX


#%% Pick Attribute that Best Splits Data

def pickAttribute(trainingData, avail_attributes):
    ### Local Variables
    data_lngth          = np.shape(trainingData)[0]
    total_attributes    = len(avail_attributes)
    attributes_infoGain = np.zeros((total_attributes,1))
        
    ### Calculate Total Entropy/GiniIndex/MajorityError
    label_ctgrs, label_cnt = np.unique(trainingData[:,total_attributes],return_counts=1)
    total_info = calcInformationGain(label_cnt, sum(label_cnt))
    
    ### Calculate Entropy/GiniIndex/MajorityError for Each Attribute
    for attrX in np.arange(0, total_attributes):
        attr_ctgrs, attr_cnt = np.unique(trainingData[:,attrX], return_counts=1)
        
        ### Create Array for Info Loss For Each Attribute's Category
        attr_ctgrs_infoLoss = np.zeros((len(attr_ctgrs), 1))
        
        ### Loop Through Each Attribute's Category
        for attr_ctgrsX in np.arange(0, len(attr_ctgrs)):
            attr_ctgrs_idx          = [i for i in range(data_lngth) if 
                              np.array_equal(trainingData[i, attrX], attr_ctgrs[attr_ctgrsX])]
            label_ctgrs, label_cnt  = np.unique(trainingData[attr_ctgrs_idx, 
                                                            total_attributes], return_counts=1)            
            
            attr_ctgrs_infoLoss[attr_ctgrsX] = calcInformationGain(
                label_cnt, attr_cnt[attr_ctgrsX]) * (attr_cnt[attr_ctgrsX]/data_lngth)
            
            
        ### Calculate Expected Value 
        attributes_infoGain[attrX] = total_info - sum(attr_ctgrs_infoLoss)
        
    ### Information Loss
    # print(labels[avail_attributes[int(np.argmax(attributes_infoGain, axis = 0))]])
    return avail_attributes[int(np.argmax(attributes_infoGain, axis = 0))]




#%% Function Containing the Algorithms

def calcInformationGain(counts, total):
    xx = 0
    length = len(counts)
    
    if algorithmType == 'Entropy':
        for idx in np.arange(0, length): 
            if counts[idx] != 0 and total != 0:
                xx = xx - (counts[idx]/total)*np.log(counts[idx]/total)
        
    elif algorithmType == 'GiniIndex':
        for idx in np.arange(0, length): 
            if total != 0:
                xx = xx + (counts[idx]/total)**2
        xx = 1 - xx
        
    elif algorithmType == 'MajorityError':
        if len(counts) == 1:
            xx = 0
        else:
            max = int(np.argmax(counts, axis = 0))
            xx = (sum(counts) - max) / sum(counts)
            
    else:
        sys.exit('Incorrect Algorithm Type')
        
    return xx




#%% Return the Available and Used Attributes

def whichAttributes(decisionTree_attr, branchX):
    used_attributes = np.empty([0,0])
    
    ### Loop Through Each Column of the Decision Tree
    for rows in decisionTree_attr[:,branchX]:
        ### If the Variable is Empty, Skip It. The Branch Has Reached Its End
        if rows == '':
            continue

        idx = labels.index(rows)
        used_attributes = np.append(used_attributes, idx)
    
    ### Create Available Attributes Array
    used_attributes     = used_attributes.astype(int)
    avail_attributes    = np.arange(0, len(labels)-1)
    avail_attributes    = np.delete(avail_attributes, used_attributes)
        
    return used_attributes, avail_attributes
        



#%% Determine If Another Node Is Needed

def needAnotherNode(trainData, used_attributes, decisionTree_ctgr):
    ### Variable Presets
    data_lngth = np.shape(trainData)[0]
    decisionTree_ctgr = decisionTree_ctgr[decisionTree_ctgr != '']
    
    ### Determine Count of Endings For Each Current Branch
    decision_branch_idx = [i for i in range(data_lngth) if 
                              np.array_equal(trainData[i, used_attributes], decisionTree_ctgr)]
    outcome_ctgrs, outcome_cnt = np.unique(
            trainData[decision_branch_idx,len(labels)-1], return_counts=1)
    
    ### Return True if More Branches are Needed
    if len(outcome_cnt) > 1:
        return True
    else:
        return False




#%% Determine Most Likely Outcome For Decision Tree Branch

def mostLikelyOutcome(decisionTree_attr, decisionTree_ctgr, trainData, testData):
    ### Preset Variables
    data_lngth = np.shape(trainData)[0]
    dtOutcome_train = np.zeros([np.shape(trainData)[0],1], dtype=object)
    dtOutcome_test = np.zeros([np.shape(testData)[0],1], dtype=object)
    
    for idx in range(0, np.shape(decisionTree_attr)[1]):
        ## Calculate the Most Likely Outcome
        used_attributes, avail_attributes = whichAttributes(decisionTree_attr, idx)
        decisionTree_ctgrX = decisionTree_ctgr[:,idx]
        
        decision_branch_idx = [i for i in range(data_lngth) if 
                              np.array_equal(trainData[i, used_attributes], decisionTree_ctgrX[decisionTree_ctgrX != ''])]
        outcome_ctgrs, outcome_cnt = np.unique(
            trainData[decision_branch_idx,len(labels)-1], return_counts=1)
        
        dt_test_idx = [i for i in range(np.shape(testData)[0]) if 
                              np.array_equal(testData[i, used_attributes], decisionTree_ctgrX[decisionTree_ctgrX != ''])]
        
        if len(outcome_cnt) == 0:
            dtOutcome_train[decision_branch_idx] = ''
            dtOutcome_test[dt_test_idx] = ''
            # dtOutcome = np.concatenate([dtOutcome, np.array('', ndmin=1)])
        else:
            dtOutcome_train[decision_branch_idx] = outcome_ctgrs[int(np.argmax(outcome_cnt, axis = 0))]
            dtOutcome_test[dt_test_idx] = outcome_ctgrs[int(np.argmax(outcome_cnt, axis = 0))]
            # dtOutcome = np.concatenate([dtOutcome, np.array(outcome_ctgrs[int(np.argmax(outcome_cnt, axis = 0))], ndmin=1)])
        
    return dtOutcome_train, dtOutcome_test




#%% Calculate Average Prediction Error

# def avgPredictionError(trainData, decisionTree_attr, decisionTree_ctgr, dtOutcome):
#     data_lngth = np.shape(trainData)[0]
#     total_attributes = np.shape(trainData)[1]-1
#     branches = np.shape(decisionTree_attr)[1]
#     errors = 0
    
#     for idx in range(0, branches):
#         dt_ctgr_branch = decisionTree_ctgr[:,idx]
        
#         used_attributes, avail_attributes = whichAttributes(decisionTree_attr, idx)
        
#         attr_ctgrs_idx          = [i for i in range(data_lngth) if 
#                           np.array_equal(trainData[i, used_attributes], dt_ctgr_branch)]
        
#         label_ctgrs, label_cnt  = np.unique(trainData[attr_ctgrs_idx, 
#                                                         total_attributes], return_counts=1)
        
#         errors += sum(label_cnt[np.where(label_ctgrs != dtOutcome[:,idx])])
        
    
#     return errors/data_lngth




#%% MAIN
main()



