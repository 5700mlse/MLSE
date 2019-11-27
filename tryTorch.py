#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch.utils.data import Dataset, DataLoader


# In[2]:


import MLSE_Class as MLSE


# In[3]:


X_train_eclipse,Y_train_bugs_eclipse,X_test_eclipse,Y_test_bugs_eclipse = MLSE.classification_binary("differentSoftware/eclipse/eclipse_all_data.csv",
                                                                     'numberOfBugsFoundUntil',"wmc",0,
                                                                       "bugs")


# In[6]:


from imblearn.combine import SMOTEENN
from collections import Counter
smote_enn = SMOTEENN(random_state=0)
X_resampled, y_resampled = smote_enn.fit_resample(X_train_eclipse, Y_train_bugs_eclipse)
print(f'Y_resampled: {sorted(Counter(y_resampled).items())}')


# In[7]:


class ImageDataset(Dataset):
    def __init__(self, file_list, target_list):
        self.file_list = file_list
        self.target_list = target_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = self.file_list[index]
        label = self.target_list[index]
        return img, label


# In[10]:


import torch
import numpy as np


# In[12]:


X_train_eclipse_Y = torch.from_numpy(np.array(X_resampled, dtype=np.float32))
Y_train_bugs_eclipse_Y = torch.from_numpy(np.array(y_resampled, dtype=np.float32))
trainset_Y = ImageDataset(X_train_eclipse_Y, Y_train_bugs_eclipse_Y)


# In[13]:


X_train_eclipse = torch.from_numpy(np.array(X_train_eclipse, dtype=np.float32))
Y_train_bugs_eclipse = torch.from_numpy(np.array(Y_train_bugs_eclipse, dtype=np.float32))
trainset = ImageDataset(X_train_eclipse, Y_train_bugs_eclipse)


# In[14]:


X_test_eclipse = torch.from_numpy(np.array(X_test_eclipse, dtype=np.float32))
Y_test_bugs_eclipse = torch.from_numpy(np.array(Y_test_bugs_eclipse, dtype=np.float32))
testset = ImageDataset(X_test_eclipse, Y_test_bugs_eclipse)


# In[15]:


len(trainset.target_list)


# In[71]:


len(X_train_eclipse)


# In[16]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from collections import namedtuple
from IPython.display import Image
from torch.utils import data as Data
import torch.optim as optim
import copy
import os


# In[17]:


train_dataloader = DataLoader(trainset, batch_size=64, shuffle=True, drop_last=False)
train_dataloader_Y = DataLoader(trainset_Y, batch_size=64, shuffle=True, drop_last=False)


# In[18]:


test_dataloader = DataLoader(testset, batch_size=64, shuffle=True, drop_last=False)


# In[19]:


class Simple_MLP(nn.Module):
    def __init__(self, size_list):
        super(Simple_MLP, self).__init__()
        layers = []
        self.size_list = size_list
        for i in range(len(size_list) - 2):
            layers.append(nn.Linear(size_list[i],size_list[i+1]))
            layers.append(nn.BatchNorm1d(num_features = size_list[i+1]))
            layers.append(nn.ReLU())
#             layers.append(nn.Dropout(0.2))
        layers.append(nn.Linear(size_list[-2], size_list[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# In[20]:


cuda = torch.cuda.is_available()
print(cuda)

model = Simple_MLP([40,80,160, 16, 2])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
device = torch.device("cuda" if cuda else "cpu")
print(model)


# In[21]:


def train_epoch(model, train_loader, criterion, optimizer):
    scheduler.step()
    model.train()
    model.to(device)

    running_loss = 0.0
    
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):   
        optimizer.zero_grad()   # .backward() accumulates gradients
        data = data.to(device).float()
        target = target.to(device).long() # all data & model on same device

        outputs = model(data)
        loss = criterion(outputs, target)
        running_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    end_time = time.time()
    
    running_loss /= len(train_loader)
    print('Training Loss: ', running_loss, 'Time: ',end_time - start_time, 's')
    return running_loss


# In[35]:


from sklearn.metrics import recall_score  


# In[36]:


def test_model(model, test_loader, criterion):
    with torch.no_grad():
        model.eval()
        model.to(device)

        running_loss = 0.0
        total_predictions = 0.0
        correct_predictions = 0.0
        recall = []
        for batch_idx, (data, target) in enumerate(test_loader):   
            data = data.to(device).float()
            target = target.to(device).long()

            outputs = model(data)

            _, predicted = torch.max(outputs.data, 1)
            total_predictions += target.size(0)
            correct_predictions += (predicted == target).sum().item()
            recall.append(recall_score(target, predicted))
            loss = criterion(outputs, target).detach()
            running_loss += loss.item()


        running_loss /= len(test_loader)
        acc = (correct_predictions/total_predictions)*100.0
        print('Testing Loss: ', running_loss)
        print('Testing Accuracy: ', acc, '%')
        print('Testing Recall: ', np.mean(recall)*100, '%')
        return running_loss, acc


# In[37]:


cuda = torch.cuda.is_available()
print(cuda)

model = Simple_MLP([40,80,160,160,640,20,10, 2])
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)
device = torch.device("cuda" if cuda else "cpu")
print(model)


# In[38]:


n_epochs = 20
Train_loss = []
Test_loss = []
Test_acc = []
import time
for i in range(n_epochs):
    train_loss = train_epoch(model,train_dataloader_Y, criterion, optimizer)
    test_loss, test_acc = test_model(model, test_dataloader, criterion)
    Train_loss.append(train_loss)
    Test_loss.append(test_loss)
    Test_acc.append(test_acc)
    print('='*20)


# In[ ]:




