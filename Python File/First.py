#!/usr/bin/env python
# coding: utf-8

# In[4]:


def sum(row,col):
    if row == 1 or col == 1:
        return 1
    data = sum(row-1,col)+sum(row,col-1)
    print(data)
    return data
print(sum(3,3))


# In[8]:


import numpy as np
print(np.arange(10))
np.linspace(30,100,5)


# In[11]:


class Queue:
    def __init__(self):
        self.arr = [1,2,3,4,5,6,7]
    def recurse():
        if (len(self.arr)==1):
            return self.arr.pop()
        curr = self.arr.pop()
        result = self.recurse()
        self.arr.push(curr)
        return result
obj = Queue()
obj.recurse()


# In[1]:


import pandas as pd
import numpy as np


# In[3]:


mydata = pd.read_csv("CardioGoodFitness.csv")


# mydata.head()

# In[4]:


mydata.


# In[6]:


from sklearn import datasets,linear_model
dataset_X,dataset_y = datasets.load_diabetes(return_X_y=True)
print(dataset_X)
print(dataset_y)


# In[14]:


import numpy as np
datas = np.arange(10)
data_x = datas[:-5]
data_y = datas[-5:]
print(data_x,data_y)


# In[ ]:




