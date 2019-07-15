#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np

def predicted_bugNum_vs_true_bugNum(Y_test_predict,Y_test_bugs):
    n = 0
    bugrowcount = 0
    for i in range(0,len(Y_test_predict)):
        if(Y_test_bugs[i]!=False):
            bugrowcount += 1
            if(Y_test_predict[i]==Y_test_bugs[i]):
                n=n+1
#    print(n/bugrowcount,n,bugrowcount)
    return n/bugrowcount,n

def load_one_version_method_matrix(matrix_path):
    print(matrix_path)
    return pd.read_csv(matrix_path)

def get_selected_rows(columncombation,X_data):
    flag = False
    for i in columncombation:
        if(flag == False):
            c = X_data[:,i].T
            flag = True
        else:
            c = np.vstack((c,X_data[:,i].T))
    X_train_selected = c.T 
    return X_train_selected

def classification_binary(csvpath,startColumnName,
                          endColumnName,isdefects,importance_result,
                         columncombation_j = None):

    filePath = "AlltheMatrix/"

    filePathNowusing = filePath + csvpath

    changed_matrix = load_one_version_method_matrix(filePathNowusing)
    print(f'lens: {len(changed_matrix.columns.values)}')
    matrixlen = int(len(changed_matrix))
    startindex = 0
    endindex = 0
    for i, column in enumerate(changed_matrix.columns.values):
        if(column == startColumnName):
            startindex = i
        elif (column == endColumnName):
            endindex = i
    print(f'start:{startindex},end:{endindex}') 
    target = changed_matrix[endColumnName].values
    data = changed_matrix.values[:,startindex+1:endindex]
    print(target.shape)
    print(data.shape)

    #shuffleIndex
    import numpy as np
    shuffle_index = np.random.permutation(matrixlen)
    target, data = target[shuffle_index], data[shuffle_index]

    #shuffleIndex
    import numpy as np
    shuffle_index = np.random.permutation(matrixlen)
    target, data = target[shuffle_index], data[shuffle_index]
    print(type(data))
    #split the train set and the test set
    X_train = data[:round(0.7*matrixlen)]
    X_test = data[round(0.7*matrixlen):]
    Y_train = target[:round(0.7*matrixlen)]
    Y_test = target[round(0.7*matrixlen):]
    print(X_train,Y_train.shape)
    #train a binary classifier
    Y_train_bugs = (Y_train > isdefects)
    Y_test_bugs = (Y_test > isdefects)
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    if(columncombation_j!=None):
        print("####################################")
        X_train = get_selected_rows(columncombation_j,X_train)
        X_test = get_selected_rows(columncombation_j,X_test)

    forest_clf = RandomForestClassifier(n_estimators = 10)
    forest_clf.fit(X_train,Y_train_bugs)

    cross_result = cross_val_score(forest_clf, X_train, Y_train_bugs, cv = 5, scoring = 'accuracy')
    Y_forest_predict = forest_clf.predict(X_train)

    from sklearn.metrics import precision_score, recall_score
    forst_precision = precision_score(Y_train_bugs, Y_forest_predict)
    forst_recall = recall_score(Y_train_bugs, Y_forest_predict)
    print(f'Forest_accurancy is {forst_precision}')
    print(f'Forest_recall is {forst_recall}')
    print(cross_result)
    #    print(X_train_selected)
    Y_test_predict = forest_clf.predict(X_test)

    findingRate = predicted_bugNum_vs_true_bugNum(Y_test_predict,Y_test_bugs)
    print(f'fingdingrate = {findingRate}')
    
    # get the importance of each column
    importances=forest_clf.feature_importances_
    print('the importance of the each column：\n',importances)
    indices = np.argsort(importances)[::-1]
    print('the descend order for importance of the column：\n',indices)
    most_import = indices[:6]
    print(f'most importtant columns:{most_import}')
    print(X_train[:,most_import])
    

    columnlist = np.arange(0,endindex-startindex-1,1)
    for i in columnlist:
        if i in most_import:
            importance_result[i] +=1

#     import csv
#     # have to change the output file name
#     with open('PredictedAlltheMatrix/importance_result.csv','a+') as f:
#         csv_write = csv.writer(f)
#         data_row = [most_import,forst_precision,forst_recall,findingRate,np.mean(cross_result)]
#         csv_write.writerow(data_row)


# In[44]:


importance_result = {}
columnlist = np.arange(0,36,1)
for i in columnlist:
    importance_result[i] = 0
for i in range(30):
    classification_binary('all_data.csv','numberOfVersionsUntil','bugs',0,importance_result)
importance_result


# In[48]:


topTenImportantColumn = sorted(importance_result.items(),key = lambda item:item[1])[-6:]


# In[49]:


topTenImportantColumn


# In[50]:


tup = ()
for i,one in enumerate(topTenImportantColumn):
    tup += (one[0],)
print(tup)
classification_binary('all_data.csv','numberOfVersionsUntil',
                      'bugs',0,importance_result,tup)


# In[ ]:




