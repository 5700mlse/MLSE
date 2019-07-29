import pandas as pd
import alamopy
import numpy as np
import matplotlib.pyplot as plt
import json
import pickle
def get_the_predicted_bugs():
    changed_buggy_matrix = pd.read_csv("predicted_bugs.csv")
    Y_target = changed_buggy_matrix['bugs'].values
    X_data = changed_buggy_matrix.values[:,3:18]
    return [Y_target, X_data]



def grid_search():
    result = get_the_predicted_bugs()
    Y_target = result[0]
    X_data = result[1]

    alamopy.addCustomConstraints(["1 -z1"])
    R2_model_dict = {}
    doc = open('grid_test/out.txt','w')
    for i1 in range(0,3):
        for i2 in range (0,3):
            for i3 in range (0,3):
                res = alamopy.alamo(X_data,Y_target,linfcns = 1, expfcns = 1,logfcns = 1, sinfcns = 1,cosfcns = 1,monomialpower=(i1,i2,i3),multi2power=(i1,i2,i3),multi3power=(i1,i2,i3),ratiopower = (i1,i2,i3))
                del res['f(model)']
                key = str(i1)+str(i2)+str(i3)
                R2_model_dict[key] = res

                
    output = open('dict.pickle', 'wb')
    pickle.dump(R2_model_dict,output)
    output.close

    # print(R2_model_dict,file=doc)
    # doc.close()
    print(res['model'])
    print(res['R2'])
            
def load_the_pickle():
    pickle_in = open("dict.pickle","rb")
    example_dict = pickle.load(pickle_in)
    return example_dict
    
def plot_the_predict_vs_target(res):
    xline = np.linspace(0,162,162)

    model = res['f(model)']
    Y_trans_result = []
    for x_one in X_data:
        Y_trans_result += [model(x_one)]

    plt.plot(xline,Y_trans_result,'r--',label = "alamo_result")
    plt.plot(xline,Y_target,label = "target")
    plt.legend()
    plt.show()
    print('finish')


# grid_search()
# file = open('out.txt', 'r') 
# js = file.read()
# dic = json.loads(js)   
# print(dic) 
#  file.close() 
# fr = open("out.txt",'r+')
# dic = eval(fr.read())   #读取的str转换为字典
# print(dic)
# fr.close()
result = get_the_predicted_bugs()
Y_target = result[0]
X_data = result[1]
alamopy.addCustomConstraints(["1 -z1"])
res = alamopy.alamo(X_data,Y_target,linfcns = 1, expfcns = 1,logfcns = 1, sinfcns = 1,cosfcns = 1)
plot_the_predict_vs_target(res)
