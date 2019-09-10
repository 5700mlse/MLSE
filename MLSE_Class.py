#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score

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
                          endColumnName,isdefects,
                         bugcolumnName,columncombation_j = None):

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
    target = changed_matrix[bugcolumnName].values
    data = changed_matrix.values[:,startindex+1:endindex]
    print(target.shape)
    print(f'the data shape is {data.shape}')

    #shuffleIndex
    import numpy as np
    shuffle_index = np.random.permutation(matrixlen)
    target, data = target[shuffle_index], data[shuffle_index]

    print(type(data))
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
    #split the train set and the test set
    X_train = data[:round(0.7*matrixlen)]
    X_test = data[round(0.7*matrixlen):]
    Y_train = target[:round(0.7*matrixlen)]
    Y_test = target[round(0.7*matrixlen):]
    print(X_train,Y_train.shape)
    print("@@@@@@@@@@@@@@@@@@@@@@@@@@")
    #train a binary classifier
    Y_train_bugs = (Y_train > isdefects)
    Y_test_bugs = (Y_test > isdefects)

    if(columncombation_j!=None):
        print("####################################")
        X_train = get_selected_rows(columncombation_j,X_train)
        X_test = get_selected_rows(columncombation_j,X_test)
    return X_train,Y_train_bugs,X_test,Y_test_bugs

from sklearn.metrics import accuracy_score  
from sklearn.metrics import recall_score  
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from collections import Counter

def get_all_scores(y_real, y_pre):
    return accuracy_score(y_real, y_pre), recall_score(y_real, y_pre), precision_score(y_real, y_pre), f1_score(y_real, y_pre)

def append_socre_to_list(accuracy_score_list,recall_score_list,precision_score_list,f1_score_list,dataset):
    accuracy_score_list.append(dataset[0])
    recall_score_list.append(dataset[1])
    precision_score_list.append(dataset[2])
    f1_score_list.append(dataset[3])
    
def print_all_train_results(accuracy_score_list,recall_score_list,precision_score_list,f1_score_list):
    print(f'The train  accuracy score is {np.mean(accuracy_score_list)}')
    print(f'The train recall score is {np.mean(recall_score_list)}')
    print(f'The train precision score is {np.mean(precision_score_list)}')
    print(f'The train f1 score is {np.mean(f1_score_list)}')

def print_all_test_results(accuracy_score_list,recall_score_list,precision_score_list,f1_score_list):
    print(f'The test accuracy score is {np.mean(accuracy_score_list)}')
    print(f'The test recall score is {np.mean(recall_score_list)}')
    print(f'The test precision score is {np.mean(precision_score_list)}')
    print(f'The test f1 score is {np.mean(f1_score_list)}')
    
from sklearn.neighbors import KNeighborsClassifier
def KNN(X_train,Y_train_bugs,X_test,Y_test_bugs):
    accuracy_score_list = []
    recall_score_list = []
    precision_score_list = []
    f1_score_list = []
    accuracy_score_list2 = []
    recall_score_list2 = []
    precision_score_list2 = []
    f1_score_list2 = []
    for i in range(1):
        knn = KNeighborsClassifier(n_neighbors=3,weights = 'distance')
        knn.fit(X_train,Y_train_bugs)
        Y_train_predict = knn.predict(X_train)
        Y_test_predict = knn.predict(X_test)
        train_set_result = get_all_scores(Y_train_bugs,Y_train_predict)
        test_set_result = get_all_scores(Y_test_bugs,Y_test_predict)
        append_socre_to_list(accuracy_score_list,recall_score_list,precision_score_list,f1_score_list,train_set_result)
        append_socre_to_list(accuracy_score_list2,recall_score_list2,precision_score_list2,f1_score_list2,test_set_result)
    #print_all_train_results(accuracy_score_list,recall_score_list,precision_score_list,f1_score_list)
    print_all_test_results(accuracy_score_list2,recall_score_list2,precision_score_list2,f1_score_list2)
    
from sklearn.naive_bayes import GaussianNB
def NaivceBayes(X_train,Y_train_bugs,X_test,Y_test_bugs):
    accuracy_score_list = []
    recall_score_list = []
    precision_score_list = []
    f1_score_list = []
    accuracy_score_list2 = []
    recall_score_list2 = []
    precision_score_list2 = []
    f1_score_list2 = []
    for i in range(10):
        NB =  GaussianNB()
        NB.fit(X_train,Y_train_bugs)
        Y_train_predict = NB.predict(X_train)
        Y_test_predict = NB.predict(X_test)
        train_set_result = get_all_scores(Y_train_bugs,Y_train_predict)
        test_set_result = get_all_scores(Y_test_bugs,Y_test_predict)
        append_socre_to_list(accuracy_score_list,recall_score_list,precision_score_list,f1_score_list,train_set_result)
        append_socre_to_list(accuracy_score_list2,recall_score_list2,precision_score_list2,f1_score_list2,test_set_result)
    #print_all_train_results(accuracy_score_list,recall_score_list,precision_score_list,f1_score_list)
    print_all_test_results(accuracy_score_list2,recall_score_list2,precision_score_list2,f1_score_list2)
    
from sklearn.linear_model import LogisticRegression
def logisticRegression(X_train,Y_train_bugs,X_test,Y_test_bugs):
    accuracy_score_list = []
    recall_score_list = []
    precision_score_list = []
    f1_score_list = []
    accuracy_score_list2 = []
    recall_score_list2 = []
    precision_score_list2 = []
    f1_score_list2 = []
    for i in range(5):
        LR_clf = LogisticRegression(solver='liblinear',max_iter=1000,class_weight='balanced')
        LR_clf.fit(X_train,Y_train_bugs)
        Y_train_predict = LR_clf.predict(X_train)
        Y_test_predict = LR_clf.predict(X_test)
        train_set_result = get_all_scores(Y_train_bugs,Y_train_predict)
        test_set_result = get_all_scores(Y_test_bugs,Y_test_predict)
        append_socre_to_list(accuracy_score_list,recall_score_list,precision_score_list,f1_score_list,train_set_result)
        append_socre_to_list(accuracy_score_list2,recall_score_list2,precision_score_list2,f1_score_list2,test_set_result)
    #print_all_train_results(accuracy_score_list,recall_score_list,precision_score_list,f1_score_list)
    print_all_test_results(accuracy_score_list2,recall_score_list2,precision_score_list2,f1_score_list2)
    
from sklearn.tree import DecisionTreeClassifier
def DecisionTree(X_train,Y_train_bugs,X_test,Y_test_bugs):
    accuracy_score_list = []
    recall_score_list = []
    precision_score_list = []
    f1_score_list = []
    accuracy_score_list2 = []
    recall_score_list2 = []
    precision_score_list2 = []
    f1_score_list2 = []
    print(len(accuracy_score_list2))
    X_train,Y_train_bugs,X_test,Y_test_bugs = classification_binary("modified_eclipse-metrics-files-3.0.csv",'FOUT_avg',"NORM_SuperMethodInvocation",0,
                                   "post")
    for i in range(10):
        DT = DecisionTreeClassifier(random_state=0,class_weight='balanced')
        DT.fit(X_train,Y_train_bugs)
        Y_train_predict = DT.predict(X_train)
        Y_test_predict = DT.predict(X_test)
        train_set_result = get_all_scores(Y_train_bugs,Y_train_predict)
        test_set_result = get_all_scores(Y_test_bugs,Y_test_predict)
        append_socre_to_list(accuracy_score_list,recall_score_list,precision_score_list,f1_score_list,train_set_result)
        append_socre_to_list(accuracy_score_list2,recall_score_list2,precision_score_list2,f1_score_list2,test_set_result)
    #print_all_train_results(accuracy_score_list,recall_score_list,precision_score_list,f1_score_list)
    print_all_test_results(accuracy_score_list2,recall_score_list2,precision_score_list2,f1_score_list2)
    
from sklearn.ensemble import RandomForestClassifier  
def RandomForest(X_train,Y_train_bugs,X_test,Y_test_bugs):
    accuracy_score_list = []
    recall_score_list = []
    precision_score_list = []
    f1_score_list = []
    accuracy_score_list2 = []
    recall_score_list2 = []
    precision_score_list2 = []
    f1_score_list2 = []
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    scaler.fit(X_test)
    X_test= scaler.transform(X_test)
    for i in range(5):
        RF = RandomForestClassifier(n_estimators = 200,criterion = 'entropy')
        RF.fit(X_train,Y_train_bugs)
        Y_train_predict = RF.predict(X_train)
        Y_test_predict = RF.predict(X_test)
        train_set_result = get_all_scores(Y_train_bugs,Y_train_predict)
        test_set_result = get_all_scores(Y_test_bugs,Y_test_predict)
        append_socre_to_list(accuracy_score_list,recall_score_list,precision_score_list,f1_score_list,train_set_result)
        append_socre_to_list(accuracy_score_list2,recall_score_list2,precision_score_list2,f1_score_list2,test_set_result)
    #print_all_train_results(accuracy_score_list,recall_score_list,precision_score_list,f1_score_list)
    print_all_test_results(accuracy_score_list2,recall_score_list2,precision_score_list2,f1_score_list2)
    
from sklearn.svm import SVC
def SVM_clf(X_train,Y_train_bugs,X_test,Y_test_bugs):
    accuracy_score_list = []
    recall_score_list = []
    precision_score_list = []
    f1_score_list = []
    accuracy_score_list2 = []
    recall_score_list2 = []
    precision_score_list2 = []
    f1_score_list2 = []
    for i in range(10):
        svmclf = SVC(gamma='auto')
        svmclf.fit(X_train,Y_train_bugs)
        Y_train_predict = svmclf.predict(X_train)
        Y_test_predict = svmclf.predict(X_test)
        train_set_result = get_all_scores(Y_train_bugs,Y_train_predict)
        test_set_result = get_all_scores(Y_test_bugs,Y_test_predict)
        append_socre_to_list(accuracy_score_list,recall_score_list,precision_score_list,f1_score_list,train_set_result)
        append_socre_to_list(accuracy_score_list2,recall_score_list2,precision_score_list2,f1_score_list2,test_set_result)
    #print_all_train_results(accuracy_score_list,recall_score_list,precision_score_list,f1_score_list)
    print_all_test_results(accuracy_score_list2,recall_score_list2,precision_score_list2,f1_score_list2)
    
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler    
def neuralNetworks(X_train,Y_train_bugs,X_test,Y_test_bugs):
    accuracy_score_list = []
    recall_score_list = []
    precision_score_list = []
    f1_score_list = []
    accuracy_score_list2 = []
    recall_score_list2 = []
    precision_score_list2 = []
    f1_score_list2 = []
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    scaler.fit(X_test)
    X_test= scaler.transform(X_test)
    for i in range(5):
        mclf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 100))
        mclf.fit(X_train,Y_train_bugs)
        Y_train_predict = mclf.predict(X_train)
        Y_test_predict = mclf.predict(X_test)
        train_set_result = get_all_scores(Y_train_bugs,Y_train_predict)
        test_set_result = get_all_scores(Y_test_bugs,Y_test_predict)
        append_socre_to_list(accuracy_score_list,recall_score_list,precision_score_list,f1_score_list,train_set_result)
        append_socre_to_list(accuracy_score_list2,recall_score_list2,precision_score_list2,f1_score_list2,test_set_result)
    # print_all_train_results(accuracy_score_list,recall_score_list,precision_score_list,f1_score_list)
    print_all_test_results(accuracy_score_list2,recall_score_list2,precision_score_list2,f1_score_list2)

from imblearn.over_sampling import RandomOverSampler
from collections import Counter


# In[ ]:




