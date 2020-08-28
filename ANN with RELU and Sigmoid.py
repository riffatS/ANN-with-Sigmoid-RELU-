#!/usr/bin/env python
# coding: utf-8

# In[1]:


#sigmoid function and one layer
import numpy as np
def nonlinear(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

x=np.array([[0,0,0],
            [4,0,1],
            [0,1,1],
            [4,0,0],
            [1,1,0]])
x.shape
y=np.array([[1,4,1,4,1]]).T
y.shape

#randomly defining wiehgts they will later be updated
np.random.seed(2)
syn0=np.random.random((3,1))
syn0
for iter in  range(100000):
    layer0=x
    layer1=nonlinear(np.dot(layer0,syn0))
    l1_error=y-layer1
    l1_delta=l1_error*nonlinear(layer1,True)
    
    #weight updated
    syn0+=np.dot(layer0.T,l1_delta)
print("output after training")
print((layer1))


# In[2]:


#sigmoid function and multi layer
import numpy as np
def nonlinear(x, deriv=False):
    if(deriv==True):return x*(1-x)
    return 1/(1+np.exp(-x))
    

x=np.array([[0,0,0],
            [4,0,1],
            [0,1,1],
            [4,0,0],
            [1,1,0]])
x.shape
y=np.array([[1,4,1,4,1]]).T
y.shape

#randomly defining wiehgts they will later be updated
np.random.seed(2)
syn0=np.random.random((3,5))
syn1=np.random.random((5,4))
syn2=np.random.random((4,1))
# print(syn0)
for iter in  range(600000):
    layer0=x
    layer1=nonlinear(np.dot(layer0,syn0))
    layer2=nonlinear(np.dot(layer1,syn1))
    layer3=nonlinear(np.dot(layer2,syn2))
    l2_error=y-layer2
    l3_error=y-layer3
    l3_delta=l3_error*nonlinear(layer3,deriv=True)
    l2_delta=l2_error*nonlinear(layer2,deriv=True)
    l1_error=l2_delta.dot(syn1.T)
    l1_delta=l1_error*nonlinear(layer1,deriv=True)
    
    #weight updated
    syn0+=np.dot(layer0.T,l1_delta)
    syn1+=np.dot(layer1.T,l2_delta)
    syn2+=np.dot(layer2.T,l3_delta)
print("output after training")
print((layer3))


# In[8]:


#RELU function and MULTI layer
import numpy as np
def rectified(x,deriv=False):
    if(deriv==True):
        return x if x.all()>0 else 0
    return x
    

x=np.array([
            [0,0,0],
            [4,0,1],
            [0,1,1],
            [4,0,0],
            [1,1,0]])
x.shape
y=np.array([[1,4,1,4,1]]).T
y.shape

#randomly defining wiehgts they will later be updated
np.random.seed(4)
syn0=np.random.random((3,5))
# print(syn0)
syn1=np.random.random((5,4))
syn2=np.random.random((4,1))
# print(syn0)
for iter in  range(90000):
    layer0=x
    layer1=rectified(np.dot(layer0,syn0))
    layer2=rectified(np.dot(layer1,syn1))
    layer3=rectified(np.dot(layer2,syn2))
    l2_error=y-layer2
    l3_error=y-layer3
    l3_delta=l3_error*rectified(layer3,deriv=True)
    l2_delta=l2_error*rectified(layer2,deriv=True)
    l1_error=l2_delta.dot(syn1.T)
    l1_delta=l1_error*rectified(layer1,deriv=True)
    
    #weight updated
    syn0+=np.dot(layer0.T,l1_delta)
    syn1+=np.dot(layer1.T,l2_delta)
    syn2+=np.dot(layer2.T,l3_delta)
print("output after training")
print((layer1))


# In[ ]:





# In[9]:


#RELU function and one layer
import numpy as np
def rectified(x,deriv=False):
    if(deriv==True):
        return x if x.all()>0 else 0
    return x
    
x=np.array([[0,0,0],
            [4,0,1],
            [0,1,1],
            [4,0,0],
            [1,1,0]])
x.shape
y=np.array([[1,4,1,4,1]]).T
y.shape

#randomly defining wiehgts they will later be updated
np.random.seed(4)
syn0=np.random.random((3,1))
syn0
for iter in  range(10000000):
    layer0=x
    layer1=rectified(np.dot(layer0,syn0))
    l1_error=y-layer1
    l1_delta=l1_error*rectified(layer1,True)
    
    #weight updated
    syn0+=np.dot(layer0.T,l1_delta)
print("output after training")
print((layer1))

