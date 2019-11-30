#!/usr/bin/env python
# coding: utf-8

# In[1]:

#!/usr/bin/env python
# coding: utf-8

# # step 1: prepare the dataset

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


# In[2]:


X_train,Y_train_bugs,X_test,Y_test_bugs = classification_binary("modified_eclipse-metrics-files-3.0.csv",'FOUT_avg',"NORM_SuperMethodInvocation",0,
                               "post")


# ## The dataset is clearly imbalanced, and I am going to use the following 5 methods to deal with it
# 

# # step 2: five ways to deal with the imbalanced dataset
# 

# In[34]:


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
    return [np.mean(accuracy_score_list),np.mean(recall_score_list),np.mean(precision_score_list),np.mean(f1_score_list)]

def print_all_test_results(accuracy_score_list,recall_score_list,precision_score_list,f1_score_list):
    print(f'The test accuracy score is {np.mean(accuracy_score_list)}')
    print(f'The test recall score is {np.mean(recall_score_list)}')
    print(f'The test precision score is {np.mean(precision_score_list)}')
    print(f'The test f1 score is {np.mean(f1_score_list)}')
    return [np.mean(accuracy_score_list),np.mean(recall_score_list),np.mean(precision_score_list),np.mean(f1_score_list)]


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

# In[42]:


import xlrd
import xlwt
from xlutils.copy import copy
 
def write_excel_xls(path, sheet_name, value):
    index = len(value)
    workbook = xlwt.Workbook()  
    sheet = workbook.add_sheet(sheet_name) 
    for i in range(0, index):
        for j in range(0, len(value[i])):
            sheet.write(i, j, value[i][j]) 
    workbook.save(path)  
    print("write Sucess")
 
 
def write_excel_xls_append(path, value):
    index = len(value)  
    workbook = xlrd.open_workbook(path)
    sheets = workbook.sheet_names() 
    worksheet = workbook.sheet_by_name(sheets[0]) 
    rows_old = worksheet.nrows  
    new_workbook = copy(workbook)
    new_worksheet = new_workbook.get_sheet(0)
    for i in range(0, index):
        for j in range(0, len(value[i])):
            new_worksheet.write(i+rows_old, j, value[i][j]) 
    new_workbook.save(path)  
    print("append sucess")

filename=xlwt.Workbook()  

sheet=filename.add_sheet("test")   
filename.save("./test1.xls")   
value = [['method','accuracy','recall','precision','f1']]
write_excel_xls_append('test1.xls', value)


# In[44]:


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
    res = print_all_test_results(accuracy_score_list2,recall_score_list2,precision_score_list2,f1_score_list2)
    return res
    
value = [['KNN']+KNN(X_train,Y_train_bugs,X_test,Y_test_bugs)]
write_excel_xls_append('test1.xls', value)


# In[40]:


value


# # Naïve Bayes,

# In[46]:


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
    return print_all_test_results(accuracy_score_list2,recall_score_list2,precision_score_list2,f1_score_list2)

value = [['NaiveBayes']+NaivceBayes(X_train,Y_train_bugs,X_test,Y_test_bugs)]
write_excel_xls_append('test1.xls', value)


# # Logistic Regression

# In[47]:


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
    return print_all_test_results(accuracy_score_list2,recall_score_list2,precision_score_list2,f1_score_list2)

value = [['logisticRegression']+logisticRegression(X_train,Y_train_bugs,X_test,Y_test_bugs)]
write_excel_xls_append('test1.xls', value)


# # Decision Tree,

# In[48]:


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
    return print_all_test_results(accuracy_score_list2,recall_score_list2,precision_score_list2,f1_score_list2)

value = [['DecisionTree']+DecisionTree(X_train,Y_train_bugs,X_test,Y_test_bugs)]
write_excel_xls_append('test1.xls', value)


# # Random Forest

# In[56]:


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
    return print_all_test_results(accuracy_score_list2,recall_score_list2,precision_score_list2,f1_score_list2)

value = [['RamdonForest']+RandomForest(X_train,Y_train_bugs,X_test,Y_test_bugs)]
write_excel_xls_append('test1.xls', value)


# # SVM

# In[50]:


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
#SVM_clf(X_train,Y_train_bugs,X_test,Y_test_bugs)


# # Neural Networks

# In[57]:


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
    return print_all_test_results(accuracy_score_list2,recall_score_list2,precision_score_list2,f1_score_list2)
neuralNetworks(X_train,Y_train_bugs,X_test,Y_test_bugs)

value = [['neuralNetworks']+neuralNetworks(X_train,Y_train_bugs,X_test,Y_test_bugs)]
write_excel_xls_append('test1.xls', value)


# # step 2.2: oversample minority class

# In[58]:


from imblearn.over_sampling import RandomOverSampler
from collections import Counter
ros = RandomOverSampler(random_state=0)


# In[59]:


X_train,Y_train_bugs,X_test,Y_test_bugs = classification_binary("modified_eclipse-metrics-files-3.0.csv",'FOUT_avg',"NORM_SuperMethodInvocation",0,
                               "post")
X_resampled, y_resampled = ros.fit_resample(X_train, Y_train_bugs)
print(f'Y_resampled: {sorted(Counter(y_resampled).items())}')


# # step 2.2 K-nearest neighbors

# In[60]:


Over = 'OverSample_'
value = [[Over+'KNN']+KNN(X_resampled,y_resampled,X_test,Y_test_bugs)]
write_excel_xls_append('test1.xls', value)


# # step 2.2 Naïve Bayes,

# In[61]:


value = [[Over+'NavieBayes']+NaivceBayes(X_resampled,y_resampled,X_test,Y_test_bugs)]
write_excel_xls_append('test1.xls', value)


# # step 2.2 Logistic Regression

# In[62]:


X_train,Y_train_bugs,X_test,Y_test_bugs = classification_binary("modified_eclipse-metrics-files-3.0.csv",'FOUT_avg',"StringLiteral",0,
                               "post")
X_resampled, y_resampled = ros.fit_resample(X_train, Y_train_bugs)

value = [[Over+'logisticRegression']+logisticRegression(X_resampled,y_resampled,X_test,Y_test_bugs)]
write_excel_xls_append('test1.xls', value)


# # step 2.2 Decision Tree

# In[63]:


X_train,Y_train_bugs,X_test,Y_test_bugs = classification_binary("modified_eclipse-metrics-files-3.0.csv",'FOUT_avg',"NORM_SuperMethodInvocation",0,
                               "post")
X_resampled, y_resampled = ros.fit_resample(X_train, Y_train_bugs)

value = [[Over+'DecisionTree']+DecisionTree(X_resampled,y_resampled,X_test,Y_test_bugs)]
write_excel_xls_append('test1.xls', value)


# # step 2.2 Random Forest

# In[64]:



value = [[Over+'RandomForest']+RandomForest(X_resampled,y_resampled,X_test,Y_test_bugs)]
write_excel_xls_append('test1.xls', value)


# # step 2.2 Neural Networks

# In[65]:



value = [[Over+'neuralNetworks']+neuralNetworks(X_resampled,y_resampled,X_test,Y_test_bugs)]
write_excel_xls_append('test1.xls', value)


# # step 2.2: Undersample majority class

# In[66]:


from imblearn.under_sampling import ClusterCentroids
cc = ClusterCentroids(random_state=0)
X_train,Y_train_bugs,X_test,Y_test_bugs = classification_binary("modified_eclipse-metrics-files-3.0.csv",'FOUT_avg',"NORM_SuperMethodInvocation",0,
                               "post")
print(len(X_resampled[0]),len(X_test[0]))
X_resampled, y_resampled = cc.fit_resample(X_train, Y_train_bugs)
print(f'Y_resampled: {sorted(Counter(y_resampled).items())}')


# In[67]:


len(X_resampled[0]),len(X_test[0])


# In[68]:


under = 'Under_Sample'

value = [[under+'KNN']+KNN(X_resampled,y_resampled,X_test,Y_test_bugs)]
write_excel_xls_append('test1.xls', value)


# In[69]:



value = [[under+'NaiveBayes']+NaivceBayes(X_resampled,y_resampled,X_test,Y_test_bugs)]
write_excel_xls_append('test1.xls', value)


# In[70]:




value = [[under+'LogisticRegression']+logisticRegression(X_resampled,y_resampled,X_test,Y_test_bugs)]
write_excel_xls_append('test1.xls', value)


# In[71]:



value = [[under+'DecisionTree']+DecisionTree(X_resampled,y_resampled,X_test,Y_test_bugs)]
write_excel_xls_append('test1.xls', value)


# In[72]:



value = [[under+'RandomFOrest']+RandomForest(X_resampled,y_resampled,X_test,Y_test_bugs)]
write_excel_xls_append('test1.xls', value)


# In[73]:



value = [[under+'NeuralNetworks']+neuralNetworks(X_resampled,y_resampled,X_test,Y_test_bugs)
]
write_excel_xls_append('test1.xls', value)


# # step 2.3: Generate synthetic samples

# In[74]:


from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_train,Y_train_bugs,X_test,Y_test_bugs = classification_binary("modified_eclipse-metrics-files-3.0.csv",'FOUT_avg',"NORM_SuperMethodInvocation",0,
                               "post")
X_resampled, y_resampled = smote_enn.fit_resample(X_train, Y_train_bugs)
print(f'Y_resampled: {sorted(Counter(y_resampled).items())}')


# In[75]:


under = 'Syn_Sample'

value = [[under+'KNN']+KNN(X_resampled,y_resampled,X_test,Y_test_bugs)]
write_excel_xls_append('test1.xls', value)


# In[76]:



value = [[under+'NaiveBayes']+NaivceBayes(X_resampled,y_resampled,X_test,Y_test_bugs)]
write_excel_xls_append('test1.xls', value)


# In[77]:




value = [[under+'LogisticRegression']+logisticRegression(X_resampled,y_resampled,X_test,Y_test_bugs)]
write_excel_xls_append('test1.xls', value)


# In[78]:



value = [[under+'DecisionTree']+DecisionTree(X_resampled,y_resampled,X_test,Y_test_bugs)]
write_excel_xls_append('test1.xls', value)


# In[79]:



value = [[under+'RandomFOrest']+RandomForest(X_resampled,y_resampled,X_test,Y_test_bugs)]
write_excel_xls_append('test1.xls', value)


# In[80]:


value = [[under+'NeuralNetworks']+neuralNetworks(X_resampled,y_resampled,X_test,Y_test_bugs)]
write_excel_xls_append('test1.xls', value)


# In[ ]:






from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import pandas as pd
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score, recall_score


# In[2]:


import os


import numpy as np
from scipy.stats import norm

from matplotlib import pyplot

np.random.seed(3)

num_per_class = 40
#生成样本
X = np.hstack((norm.rvs(2, size=num_per_class, scale=2),
              norm.rvs(8, size=num_per_class, scale=3)))
y = np.hstack((np.zeros(num_per_class),
               np.ones(num_per_class)))


def lr_model(clf, X):
    return 1.0 / (1.0 + np.exp(-(clf.intercept_ + clf.coef_ * X)))

from sklearn.linear_model import LogisticRegression
logclf = LogisticRegression()
print(logclf)
logclf.fit(X.reshape(num_per_class * 2, 1), y)
print(np.exp(logclf.intercept_), np.exp(logclf.coef_.ravel()))
print("P(x=-1)=%.2f\tP(x=7)=%.2f" %
      (lr_model(logclf, -1), lr_model(logclf, 7)))
X_test = np.arange(-5, 20, 0.1)
pyplot.figure(figsize=(10, 4))
pyplot.xlim((-5, 20))
pyplot.scatter(X, y, c=y)
pyplot.xlabel("feature value")
pyplot.ylabel("class")
pyplot.grid(True, linestyle='-', color='0.75')

def lin_model(clf, X):
    return clf.intercept_ + clf.coef_ * X

from sklearn.linear_model import LinearRegression
clf = LinearRegression()
print(clf)
clf.fit(X.reshape(num_per_class * 2, 1), y)
X_odds = np.arange(0, 1, 0.001)
pyplot.figure(figsize=(10, 4))
pyplot.subplot(1, 2, 1)
pyplot.scatter(X, y, c=y)
pyplot.plot(X_test, lin_model(clf, X_test))
pyplot.xlabel("feature value")
pyplot.ylabel("class")
pyplot.title("linear fit on original data")
pyplot.grid(True, linestyle='-', color='0.75')

X_ext = np.hstack((X, norm.rvs(20, size=100, scale=5)))
y_ext = np.hstack((y, np.ones(100)))
clf = LinearRegression()
clf.fit(X_ext.reshape(num_per_class * 2 + 100, 1), y_ext)
pyplot.subplot(1, 2, 2)
pyplot.scatter(X_ext, y_ext, c=y_ext)
pyplot.plot(X_ext, lin_model(clf, X_ext))
pyplot.xlabel("feature value")
pyplot.ylabel("class")
pyplot.title("linear fit on additional data")
pyplot.grid(True, linestyle='-', color='0.75')


pyplot.figure(figsize=(10, 4))
pyplot.xlim((-5, 20))
pyplot.scatter(X, y, c=y)
pyplot.plot(X_test, lr_model(logclf, X_test).ravel())
pyplot.plot(X_test, np.ones(X_test.shape[0]) * 0.5, "--")
pyplot.xlabel("feature value")
pyplot.ylabel("class")
pyplot.grid(True, linestyle='-', color='0.75')


X = np.arange(0, 1, 0.001)
pyplot.figure(figsize=(10, 4))
pyplot.subplot(1, 2, 1)
pyplot.xlim((0, 1))
pyplot.ylim((0, 10))
pyplot.plot(X, X / (1 - X))
pyplot.xlabel("P")
pyplot.ylabel("odds = P / (1-P)")
pyplot.grid(True, linestyle='-', color='0.75')

pyplot.subplot(1, 2, 2)
pyplot.xlim((0, 1))
pyplot.plot(X, np.log(X / (1 - X)))
pyplot.xlabel("P")
pyplot.ylabel("log(odds) = log(P / (1-P))")
pyplot.grid(True, linestyle='-', color='0.75')


# In[3]:


import MLSE_Class as MLSE
X_train_eclipse,Y_train_bugs_eclipse = MLSE.classification_binary2("differentSoftware/eclipse/eclipse_all_data.csv",
                                                                     'numberOfBugsFoundUntil',"wmc",0,
                                                                       "bugs")
X_train_equinox,Y_train_bugs_equinox= MLSE.classification_binary2("differentSoftware/equinox/equinox_all_data.csv",
                                                                     'numberOfBugsFoundUntil',"wmc",0,
                                                                       "bugs")
X_train_pde,Y_train_bugs_pde = MLSE.classification_binary2("differentSoftware/pde/pde_all_data.csv",
                                                                     'numberOfBugsFoundUntil',"wmc",0,
                                                                       "bugs")
X_train_lucene,Y_train_bugs_lucene = MLSE.classification_binary2("differentSoftware/lucene/lucene_all_data.csv",
                                                                     'numberOfBugsFoundUntil',"wmc",0,
                                                                       "bugs")
X_train_mylyn,Y_train_bugs_mylyn = MLSE.classification_binary2("differentSoftware/mylyn/mylyn_all_data.csv",
                                                                     'numberOfBugsFoundUntil',"wmc",0,
                                                                       "bugs")


# In[4]:


ros = RandomOverSampler(random_state=0)
X_resampled_eclipse, y_resampled_eclipse = ros.fit_resample(X_train_eclipse, Y_train_bugs_eclipse)
X_resampled_equinox, y_resampled_equinox = ros.fit_resample(X_train_equinox, Y_train_bugs_equinox)
X_resampled_pde, y_resampled_pde = ros.fit_resample(X_train_pde, Y_train_bugs_pde)
X_resampled_lucene, y_resampled_lucene = ros.fit_resample(X_train_lucene, Y_train_bugs_lucene)
X_resampled_mylyn, y_resampled_mylyn = ros.fit_resample(X_train_mylyn, Y_train_bugs_mylyn)


# In[5]:


sorted(Counter(Y_train_bugs_eclipse).items())


# In[6]:


print(f'Y_train_bugs_eclipse:{sorted(Counter(Y_train_bugs_eclipse).items())}')
print(f'Y_train_bugs_equinox:{sorted(Counter(Y_train_bugs_equinox).items())}')
print(f'Y_train_bugs_pde:{sorted(Counter(Y_train_bugs_pde).items())}')
print(f'Y_train_bugs_lucene:{sorted(Counter(Y_train_bugs_lucene).items())}')
print(f'Y_train_bugs_mylyn:{sorted(Counter(Y_train_bugs_mylyn).items())}')


# In[7]:


sorted(Counter(Y_train_bugs_eclipse).items())[0][1]


# In[12]:



import matplotlib.pyplot as plt
num_list = [791,195,1288,627,1627]
num_list1 = [206,129,209,64,245]

def data_distribution(num_list,num_list1,name):
    name_list = ['Eclipse','equinox','pde','lucene','mylyn']
    num_list = num_list
    num_list1 = num_list1
    plt.bar(range(len(num_list)), num_list, label='data_without_bugs',fc = 'g')
    plt.bar(range(len(num_list)), num_list1, bottom=num_list, label='data_with_bugs',tick_label = name_list,fc = 'y')
    plt.xlabel("dataset'name")
    plt.ylabel("number of data")
    plt.legend()


    x = np.arange(len(name_list))
    y = np.array(num_list)
    y1 = np.array(num_list1)
    sum1 = y+y1
    for a,b in zip(x,y):
        print(a,b)
        plt.text(a-0.3,b, f'{b*100/sum1[a]:.2f}%',va = 'top')

    for a,b in zip(x,y1):
        print(a,b)
        plt.text(a-0.3,b+y[a], f'{b*100/sum1[a]:.2f}%',va = 'center')
    plt.savefig(name)
    plt.show()
data_distribution([791,195,1288,627,1627],[206,129,209,64,245],'five_original_dataset.png')


# In[13]:


print(f'Y_train_bugs_eclipse:{sorted(Counter(y_resampled_eclipse).items())}')
print(f'Y_train_bugs_eclipse:{sorted(Counter(y_resampled_equinox).items())}')
print(f'Y_train_bugs_eclipse:{sorted(Counter(y_resampled_pde).items())}')
print(f'Y_train_bugs_eclipse:{sorted(Counter(y_resampled_lucene).items())}')
print(f'Y_train_bugs_eclipse:{sorted(Counter(y_resampled_mylyn).items())}')
data_distribution([791,195,1288,627,1617],[791,195,1288,627,1617],"five_oversample_dataset")


# In[14]:


from imblearn.combine import SMOTEENN
smote_enn = SMOTEENN(random_state=0)
X_resampled_eclipse, y_resampled_eclipse = smote_enn.fit_resample(X_train_eclipse, Y_train_bugs_eclipse)
X_resampled_equinox, y_resampled_equinox = smote_enn.fit_resample(X_train_equinox, Y_train_bugs_equinox)
X_resampled_pde, y_resampled_pde = smote_enn.fit_resample(X_train_pde, Y_train_bugs_pde)
X_resampled_lucene, y_resampled_lucene = smote_enn.fit_resample(X_train_lucene, Y_train_bugs_lucene)
X_resampled_mylyn, y_resampled_mylyn = smote_enn.fit_resample(X_train_mylyn, Y_train_bugs_mylyn)
print(f'Y_train_bugs_eclipse:{sorted(Counter(y_resampled_eclipse).items())}')
print(f'Y_train_bugs_eclipse:{sorted(Counter(y_resampled_equinox).items())}')
print(f'Y_train_bugs_eclipse:{sorted(Counter(y_resampled_pde).items())}')
print(f'Y_train_bugs_eclipse:{sorted(Counter(y_resampled_lucene).items())}')
print(f'Y_train_bugs_eclipse:{sorted(Counter(y_resampled_mylyn).items())}')


# In[11]:


data_distribution([556,98,761,440,1080],[673,117,1178,583,1498],"five_Synsample_dataset")


# In[15]:


X_eclipse1,Y_bugs_eclipse1 = MLSE.classification_binary2("modified_eclipse-metrics-files-2.0.csv",
                                                                     'ACD',"NORM_InstanceofExpression",0,
                                                                       "post")
X_eclipse2,Y_bugs_eclipse2 = MLSE.classification_binary2("modified_eclipse-metrics-files-2.1.csv",
                                                                     'ACD',"NORM_InstanceofExpression",0,
                                                                       "post")
X_eclipse3,Y_bugs_eclipse3 = MLSE.classification_binary2("modified_eclipse-metrics-files-3.0.csv",
                                                                     'ACD',"NORM_InstanceofExpression",0,
                                                                       "post")


# In[16]:


print(f'Y_train_eclipse1:{sorted(Counter(Y_bugs_eclipse1).items())}')
print(f'Y_train_eclipse2:{sorted(Counter(Y_bugs_eclipse2).items())}')
print(f'Y_train_eclipse3:{sorted(Counter(Y_bugs_eclipse3).items())}')


# In[17]:


name_list = ['Eclipse2.0','Eclipse2.0','Eclipse3.0']
num_list = [5754,7034,9025]
num_list1 = [975,854,1568]
plt.figure(figsize = (8,4))
plt.bar(range(len(num_list)), num_list,width=0.5, label='data_without_bugs',fc = 'g')
plt.bar(range(len(num_list)), num_list1,width=0.5, bottom=num_list, label='data_with_bugs',tick_label = name_list,fc = 'y')
plt.xlabel("dataset's name")
plt.ylabel("number of data")
plt.legend()


x = np.arange(len(name_list))
y = np.array(num_list)
y1 = np.array(num_list1)
sum1 = y+y1
for a,b in zip(x,y):
    print(a,b)
    plt.text(a-0.1,b, f'{b*100/sum1[a]:.2f}%',va = 'top')
    
for a,b in zip(x,y1):
    print(a,b)
    plt.text(a-0.1,b+y[a], f'{b*100/sum1[a]:.2f}%',va = 'center')
plt.savefig('Eclipse_dataset.png')
plt.show()


# In[18]:


import pandas
df = pandas.read_excel('test1.xls')


# In[19]:


al_names = df['method'].values
al_accuracy = df['accuracy'].values
al_recall = df['recall'].values
al_precision = df['precision'].values
al_f1 = df['f1'].values


# In[20]:


def sub_figure_plot(a,b): 
    
    name_list = al_names[a:b]
    num_list1 = al_accuracy[a:b]
    num_list2 = al_recall[a:b]
    num_list3 = al_precision[a:b]
    num_list4 = al_f1[a:b]
    x =list(range(len(name_list)))
    total_width, n = 0.8, 4
    width = total_width / n
    
    plt.bar(x, num_list1, width=width, label='accuracy',fc = 'y')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, num_list2, width=width, label='recall',tick_label = name_list,fc = 'r')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, num_list3, width=width, label='precision',fc = 'b')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, num_list4, width=width, label='f1',fc = 'g')
    plt.legend(loc = 1,prop={'size':16})
    plt.xlabel("Algorithms' Names")
    
sub_figure_plot(0,7)


# In[21]:


plt.figure(figsize = (22,24))
plt.subplot(4,1,1)
sub_figure_plot(0,6)
plt.title('Figure 8a.Result of six ML method without data preprocessing',fontsize = 'x-large')
plt.subplot(4,1,2)
sub_figure_plot(6,12)
plt.title('Figure 8b.Result of six ML method with data Over-Sample',fontsize = 'x-large')
plt.subplot(4,1,3)
sub_figure_plot(12,18)
plt.title('Figure 8c.Result of six ML method with data Under-Sample',fontsize = 'x-large')
plt.subplot(4,1,4)
sub_figure_plot(18,24)
plt.title('Figure 8d.Result of six ML method with data Synthetic-Sample',fontsize = 'x-large')
plt.savefig('different_algorithms_within_dataset.png',bbox_inches='tight',dpi=100,pad_inches=0.0)


# In[20]:


al_names[0:7]


# In[ ]:




