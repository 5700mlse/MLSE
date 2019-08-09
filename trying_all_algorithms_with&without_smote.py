#!/usr/bin/env python
# coding: utf-8

# # step 1: prepare the dataset

# In[2]:


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


# In[3]:


X_train,Y_train_bugs,X_test,Y_test_bugs = classification_binary("modified_eclipse-metrics-files-3.0.csv",'FOUT_avg',"NORM_SuperMethodInvocation",0,
                               "post")


# ## The dataset is clearly imbalanced, and I am going to use the following 5 methods to deal with it
# 

# # step 2: five ways to deal with the imbalanced dataset
# 

# In[6]:


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

# def init_the_list(accuracy_score_list = accuracy_score_list,recall_score_list = recall_score_list,
#                   precision_score_list = precision_score_list,f1_score_list = f1_score_list,
#                  accuracy_score_list2 = accuracy_score_list2,recall_score_list2 = recall_score_list2,
#                   precision_score_list2 = precision_score_list2,f1_score_list2 = f1_score_list2):
#     print("in")
#     print(len(accuracy_score_list))
#     for i in range(len(accuracy_score_list)):
#         accuracy_score_list.pop()
#         recall_score_list.pop()
#         precision_score_list.pop()
#         f1_score_list.pop()
#     print(f'list:{len(accuracy_score_list)}')
#     for i in range(len(accuracy_score_list2)):
#         accuracy_score_list2.pop()
#         recall_score_list2.pop()
#         precision_score_list2.pop()
#         f1_score_list2.pop()
#     print(f'list2:{len(accuracy_score_list2)}')
    
    
accuracy_score_list = []
recall_score_list = []
precision_score_list = []
f1_score_list = []
accuracy_score_list2 = []
recall_score_list2 = []
precision_score_list2 = []
f1_score_list2 = []
print(f'Y_train_bugs:{sorted(Counter(Y_train_bugs).items())}')


# # step 2.1: trying different algorithms
# # K-nearest neighbors

# In[14]:


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
KNN(X_train,Y_train_bugs,X_test,Y_test_bugs)


# # Naïve Bayes,

# In[15]:


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
NaivceBayes(X_train,Y_train_bugs,X_test,Y_test_bugs)


# # Logistic Regression

# In[19]:


from sklearn.linear_model import LogisticRegression
X_train,Y_train_bugs,X_test,Y_test_bugs = classification_binary("modified_eclipse-metrics-files-3.0.csv",'FOUT_avg',"StringLiteral",0,
                               "post")
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
logisticRegression(X_train,Y_train_bugs,X_test,Y_test_bugs)


# # Decision Tree,

# In[17]:


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
DecisionTree(X_train,Y_train_bugs,X_test,Y_test_bugs)


# # Random Forest

# In[18]:


from sklearn.ensemble import RandomForestClassifier
X_train,Y_train_bugs,X_test,Y_test_bugs = classification_binary("modified_eclipse-metrics-files-3.0.csv",'FOUT_avg',"NORM_SuperMethodInvocation",0,
                                   "post")
def RandomForest(X_train,Y_train_bugs,X_test,Y_test_bugs):
    accuracy_score_list = []
    recall_score_list = []
    precision_score_list = []
    f1_score_list = []
    accuracy_score_list2 = []
    recall_score_list2 = []
    precision_score_list2 = []
    f1_score_list2 = []
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
RandomForest(X_train,Y_train_bugs,X_test,Y_test_bugs)


# # SVM

# In[188]:


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
SVM_clf(X_train,Y_train_bugs,X_test,Y_test_bugs)


# # Neural Networks

# In[7]:


from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
X_train,Y_train_bugs,X_test,Y_test_bugs = classification_binary("modified_eclipse-metrics-files-3.0.csv",'FOUT_avg',"NORM_SuperMethodInvocation",0,
                                   "post")
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
neuralNetworks(X_train,Y_train_bugs,X_test,Y_test_bugs)


# # step 2.2: oversample minority class

# In[52]:


from imblearn.over_sampling import RandomOverSampler
from collections import Counter
ros = RandomOverSampler(random_state=0)


# In[53]:


X_train,Y_train_bugs,X_test,Y_test_bugs = classification_binary("modified_eclipse-metrics-files-3.0.csv",'FOUT_avg',"NORM_SuperMethodInvocation",0,
                               "post")
X_resampled, y_resampled = ros.fit_resample(X_train, Y_train_bugs)
print(f'Y_resampled: {sorted(Counter(y_resampled).items())}')


# # step 2.2 K-nearest neighbors

# In[54]:


KNN(X_resampled,y_resampled,X_test,Y_test_bugs)


# # step 2.2 Naïve Bayes,

# In[211]:


NaivceBayes(X_resampled,y_resampled,X_test,Y_test_bugs)


# # step 2.2 Logistic Regression

# In[215]:


X_train,Y_train_bugs,X_test,Y_test_bugs = classification_binary("modified_eclipse-metrics-files-3.0.csv",'FOUT_avg',"StringLiteral",0,
                               "post")
X_resampled, y_resampled = ros.fit_resample(X_train, Y_train_bugs)
logisticRegression(X_resampled,y_resampled,X_test,Y_test_bugs)


# # step 2.2 Decision Tree

# In[216]:


X_train,Y_train_bugs,X_test,Y_test_bugs = classification_binary("modified_eclipse-metrics-files-3.0.csv",'FOUT_avg',"NORM_SuperMethodInvocation",0,
                               "post")
X_resampled, y_resampled = ros.fit_resample(X_train, Y_train_bugs)
DecisionTree(X_resampled,y_resampled,X_test,Y_test_bugs)


# # step 2.2 Random Forest

# In[217]:


RandomForest(X_resampled,y_resampled,X_test,Y_test_bugs)


# # step 2.2 Neural Networks

# In[10]:


neuralNetworks(X_resampled,y_resampled,X_test,Y_test_bugs)


# # step 2.2: Undersample majority class

# In[26]:


from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids(random_state=0)
X_train,Y_train_bugs,X_test,Y_test_bugs = classification_binary("modified_eclipse-metrics-files-3.0.csv",'FOUT_avg',"NORM_SuperMethodInvocation",0,
                               "post")
print(len(X_resampled[0]),len(X_test[0]))
X_resampled, y_resampled = cc.fit_resample(X_train, Y_train_bugs)
print(f'Y_resampled: {sorted(Counter(y_resampled).items())}')


# In[27]:


len(X_resampled[0]),len(X_test[0])


# In[28]:


KNN(X_resampled,y_resampled,X_test,Y_test_bugs)


# In[29]:


NaivceBayes(X_resampled,y_resampled,X_test,Y_test_bugs)


# In[30]:


logisticRegression(X_resampled,y_resampled,X_test,Y_test_bugs)


# In[31]:


DecisionTree(X_resampled,y_resampled,X_test,Y_test_bugs)


# In[32]:


RandomForest(X_resampled,y_resampled,X_test,Y_test_bugs)


# In[33]:


neuralNetworks(X_resampled,y_resampled,X_test,Y_test_bugs)


# # step 2.3: Generate synthetic samples

# In[34]:


from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_train,Y_train_bugs,X_test,Y_test_bugs = classification_binary("modified_eclipse-metrics-files-3.0.csv",'FOUT_avg',"NORM_SuperMethodInvocation",0,
                               "post")
X_resampled, y_resampled = smote_enn.fit_resample(X_train, Y_train_bugs)
print(f'Y_resampled: {sorted(Counter(y_resampled).items())}')


# In[36]:


KNN(X_resampled,y_resampled,X_test,Y_test_bugs)


# In[35]:


NaivceBayes(X_resampled,y_resampled,X_test,Y_test_bugs)


# In[37]:


logisticRegression(X_resampled,y_resampled,X_test,Y_test_bugs)


# In[38]:


DecisionTree(X_resampled,y_resampled,X_test,Y_test_bugs)


# In[39]:


RandomForest(X_resampled,y_resampled,X_test,Y_test_bugs)


# In[40]:


neuralNetworks(X_resampled,y_resampled,X_test,Y_test_bugs)


# In[ ]:




