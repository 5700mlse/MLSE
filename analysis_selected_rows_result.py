#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import numpy as np


# In[13]:


row_numbers = np.arange(5,16,1)
row_numbers


# In[22]:


res = pd.DataFrame(columns=('column_selected','precision','recall','finding_rate','cross_val'))

def load_one_version_method_matrix(matrix_path):
    print(matrix_path)
    return pd.read_csv(matrix_path,header=None, names = ['column_selected','precision','recall','finding_rate','cross_val'] )
filePath = "rows_selected/"
for row_number in row_numbers:
    selected_row_number = "Rows_selected_"+str(row_number)+"_result.csv"
    result = load_one_version_method_matrix(filePath+selected_row_number)
    sorted_result = result.sort_values(by="finding_rate" , ascending=False) 
    res = res.append(sorted_result[0:5], ignore_index=True)


# In[21]:


res


# # same in the Classification

# In[53]:


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


# In[55]:


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
from sklearn.model_selection import cross_val_score

#hardCopy to test the selected rows
columncombation_j = (0, 6, 8, 10, 13, 14)
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


# In[ ]:




