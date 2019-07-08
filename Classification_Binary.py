#!/usr/bin/env python
# coding: utf-8

# In[51]:


import pandas as pd
import os
filePath = "AlltheMatrix/"
single_version_file_path = filePath + 'modified_single_version-ck-oo.csv'
changed_file_path = filePath + 'modified_change_metrics.csv'
bug_matrix_path = filePath + 'modified_bug-metrics.csv' 
# complexity_code_change_path = 



filePathNowusing = changed_file_path

def load_one_version_method_matrix(matrix_path):
    print(matrix_path)
    return pd.read_csv(matrix_path)

changed_matrix = load_one_version_method_matrix(filePathNowusing)
print(f'lens: {len(changed_matrix.columns.values)}')

startindex = 0
endindex = 0
for i, column in enumerate(changed_matrix.columns.values):
    if(column == 'classname'):
        startindex = i
    elif (column == 'bugs'):
        endindex = i
print(f'start:{startindex},end:{endindex}') 
target = changed_matrix['bugs'].values
data = changed_matrix.values[:,startindex+1:endindex]
print(target.shape)
print(data.shape)

#shuffleIndex
import numpy as np
shuffle_index = np.random.permutation(997)
target, data = target[shuffle_index], data[shuffle_index]

#shuffleIndex
import numpy as np
shuffle_index = np.random.permutation(997)
target, data = target[shuffle_index], data[shuffle_index]


# In[218]:


from itertools import combinations
columnlist = np.arange(0,endindex-startindex-1,1)
print(columnlist)
columncombation = list(combinations(columnlist, 5))
type(columncombation[0])
len(columncombation)


# # Random Forest Classifier

# In[111]:


print(X_train),columncombation[0],len(X_train.T)


# In[210]:


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


# In[219]:


#split the train set and the test set
X_train = data[:700]
X_test = data[700:]
Y_train = target[:700]
Y_test = target[700:]
print(X_train,Y_train.shape)
#train a binary classifier
Y_train_bugs = (Y_train > 0)
Y_test_bugs = (Y_test > 0 )
from sklearn.ensemble import RandomForestClassifier
#test the result: cross-verification
from sklearn.model_selection import cross_val_score
cent = 0
for columncombation_j in columncombation:
    X_train_selected = get_selected_rows(columncombation_j,X_train)
    X_test_selected = get_selected_rows(columncombation_j,X_test)


    forest_clf = RandomForestClassifier(n_estimators = 10)
    forest_clf.fit(X_train_selected,Y_train_bugs)

    cross_result = cross_val_score(forest_clf, X_train_selected, Y_train_bugs, cv = 10, scoring = 'accuracy')
    Y_forest_predict = forest_clf.predict(X_train_selected)

    from sklearn.metrics import precision_score, recall_score
    forst_precision = precision_score(Y_train_bugs, Y_forest_predict)
    forst_recall = recall_score(Y_train_bugs, Y_forest_predict)
    print(f'Forest_accurancy is {forst_precision}')
    print(f'Forest_recall is {forst_recall}')

    print(columncombation_j)
#    print(X_train_selected)
    Y_test_predict = forest_clf.predict(X_test_selected)
    
    findingRate = predicted_bugNum_vs_true_bugNum(Y_test_predict,Y_test_bugs)
    print(f'fingdingrate = {findingRate}')

    import csv
    with open('Rows_selected_5_result.csv','a+') as f:
        csv_write = csv.writer(f)
        data_row = [columncombation_j,forst_precision,forst_recall,findingRate,cross_result]
        csv_write.writerow(data_row)
    print(f'#########finishing:{cent/len(columncombation)}')
    cent+=1
print('------------------------------------------ ----------------------------------------\n')


# In[38]:


columnlist = changed_matrix.columns.values
print(columnlist)
predicted_bugs_matrix = pd.DataFrame(columns=columnlist)
predicted_bugs_matrix = predicted_bugs_matrix.append(changed_matrix.iloc[0], ignore_index=True)
print(predicted_bugs_matrix)
predicted_bugs_matrix = pd.DataFrame(columns=columnlist)
for i,Y_one in enumerate(Y_forest_predict):
    if(Y_one == False):
        predicted_bugs_matrix = predicted_bugs_matrix.append(changed_matrix.iloc[i], ignore_index=True)
predicted_bugs_matrix


# In[187]:


columncombation_j = (2,3,4,5,6,7,14)
X_train_selected = get_selected_rows(columncombation_j,X_train)
X_test_selected = get_selected_rows(columncombation_j,X_test)


forest_clf = RandomForestClassifier(n_estimators = 10)
forest_clf.fit(X_train_selected,Y_train_bugs)

cross_result = cross_val_score(forest_clf, X_train_selected, Y_train_bugs, cv = 10, scoring = 'accuracy')
Y_forest_predict = forest_clf.predict(X_train_selected)

from sklearn.metrics import precision_score, recall_score
forst_precision = precision_score(Y_train_bugs, Y_forest_predict)
forst_recall = recall_score(Y_train_bugs, Y_forest_predict)
print(f'Forest_accurancy is {forst_precision}')
print(f'Forest_recall is {forst_recall}')

print(columncombation_j)
#    print(X_train_selected)
print(len(X_test_selected[0]))
Y_test_predict = forest_clf.predict(X_test_selected)

findingRate = predicted_bugNum_vs_true_bugNum(Y_test_predict,Y_test_bugs)
print(f'fingdingrate = {findingRate}')


# In[178]:


predicted_bugs_matrix.to_csv("Predicted"+filePathNowusing)


# # use alamo to regress the buggy code

# ### have troubles when using the alamo
