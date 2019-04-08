#!/usr/bin/env python
# coding: utf-8

# ## Assignment 1

# In[47]:


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from utils_assignment1 import LoadBatch, show_image, update_params
import pickle
import pprint
import seaborn as sns
from IPython.html.widgets import *

get_ipython().magic(u'matplotlib inline')

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

np.random.seed(200)


# ## Exercise 1: Training a multi-linear classifier
# 
# #### Section 1
# 
# ## Getting the Data

# In[48]:


#get the data
train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y, classes = LoadBatch('cifar-10')

#reshaping labels
train_x = train_set_x.transpose()
valid_x = valid_set_x.transpose()
test_x  = test_set_x.transpose()
train_y = train_set_y.transpose()
valid_y = valid_set_y.transpose()
test_y  = test_set_y.transpose()
print("Reshaping " +u'\u2713' )


# #### Section 2
# 
# ## Sneak peak  into the dataset 

# In[49]:


num_train = train_x.shape[1]
num_valid = valid_x.shape[1]
num_test = test_x.shape[1]
num_px = train_x.shape[0]


print ("Number of training examples: " + str(num_train))
print ("Number of valid examples: " + str(num_valid))
print ("Number of testing examples: " + str(num_test))
print ("Each image is of size: "+str(num_px))
print ("train_x shape: " + str(train_x.shape))
print ("train_y shape: " + str(train_y.shape))
print ("valid_x shape: " + str(valid_x.shape))
print ("valid_y shape: " + str(valid_y.shape))
print ("test_x shape: " + str(test_x.shape))
print ("test_y shape: " + str(test_y.shape))
print ("classes shape :"+str(classes.shape))


# ## Check out some pics

# In[50]:


index=1
show_image(train_set_x,index)
#using train_set_x keeping the show_image dimensions in mind
label=np.where(train_y[:,index]==1)
print ("It is a "+str(classes[label][0]))


# #### Section 2
# 
# 
# 
# ## Paramater Initialization

# In[62]:


#Number for images to train
N=10000
#Number of possible predictions
K=len(classes)
#Dimensions of the input image
d=num_px
#Subset of data to be trained on
X=train_x[:,:N]
Y=train_y[:,:N]
#Gausssian parameters 
mu=0
sigma=0.01
W = np.random.normal(loc=mu,scale=sigma,size=(K,d)) 
b = np.random.normal(loc=mu,scale=sigma,size=(K,1))

lambd=0
GDparams={}
#n_batch=batch_size NOT the number
GDparams['n_batch']=100
GDparams['eta']=0.001
GDparams['n_epochs']=200

#Get the shapes right
assert(X.shape == (d, N))
assert(Y.shape == (K, N))
assert(W.shape == (K, d))
assert(b.shape == (K, 1))
assert(classes.size==10)


# #### Section 3
# ## Evaluation

# In[63]:


def softmax(x):
    #Instead of using np.exp(x)/np.sum(np.exp(x))
    #I decrease the value of x with the max, 
    #it could be any number as it cancels out when equation is expanded
    #This is to avoid overflow due to exponential increase
    e_x = np.exp(x - np.max(x))
    soft= e_x / np.sum(e_x,axis=0)
    return soft

def EvaluateClassifier(X, W, b):
    """
    s = W x + b
    p = SOFTMAX (s) 
    """
    s = np.dot(W,X) + b
    P = softmax(s)
    #Dimensional check
    assert(X.shape == (d,N))
    assert(P.shape == (K,N))
    return P


# In[64]:


# N set to 100
P=EvaluateClassifier(X,W,b)
plt.rcParams['figure.figsize'] = [15, 10]
# plt.subplot(221).set_title("Probabilities")
# ax = sns.heatmap(P)
plt.subplot(222).set_title("Prediction")
predictions=(P == P.max(axis=0, keepdims=1)).astype(float)
ax = sns.heatmap(predictions)
plt.subplot(221).set_title("Ground Truth")
ax=sns.heatmap(Y)
plt.subplot(224).set_title("Correct Matches")
matches=(np.multiply(Y,predictions)).astype(float)
ax=sns.heatmap(matches)
plt.subplot(223).set_title("Differences")
matches= Y+predictions
ax=sns.heatmap(matches)
plt.show(ax)


# #### Section 4
# 
# ## Computing the cost

# In[65]:


def ComputeCost_CrossEntropy(P,Y,W):
 
    #N = P.shape[1]
    P = np.reshape(P, (K, -1))
    ground_truth = np.reshape(Y, (K, -1))
    
    product = np.multiply(Y, P).sum(axis=0)
    #cross entropy loss - Handling -log0 tending to infinity
    product[product == 0] = np.finfo(float).eps
    crossEntropyLoss= - np.log(product).sum() / N
    L2_regularization_cost=lambd * np.power(W, 2).sum()
    
    assert(Y.shape == (K,N))
    return  crossEntropyLoss + L2_regularization_cost


# In[66]:


cost=ComputeCost_CrossEntropy(P,Y,W)
print("cost :"+str(cost))


# #### Section 5
# 
# ## Accuracy of the predictions

# In[67]:


def ComputeAccuracy(P,Y):
    #instead of using function acc = ComputeAccuracy(X, y, W, b)
    # and calculating P again lets use it from the from the prev. output
    predictions=np.argmax(P,axis=0)
    groundtruth=np.argmax(Y,axis=0)
    matches=np.sum(predictions==groundtruth)
    total=len(predictions)
    accuracy=(matches/float(total))
    return accuracy


# In[68]:


accuracy=ComputeAccuracy(P,Y)
print ("Accuracy :"+ str(accuracy))


# #### Section 6
# 
# ## Computing the Gradients

# In[69]:


def ComputeGradients(X, Y, P, W, lambd):
    """ 
    G -- dJ/dZ - g for Batch
    Z -- WX
    grad_W1 -- dJ/dW
    grad_b -- dJ/db
    grad_L -- dJ/dL Regularization term
    grad_W -- dJ/dW + Regularization term

    lambd - lambda, to differ from the keyword itself
    """
    N=float(X.shape[1])
    G= -1*(Y-P)
    grad_W1= (1/N)*np.dot(G,X.T)
    grad_L = 2*lambd*W
    grad_W = grad_W1+grad_L
    grad_b = (1/N)*np.dot(G,np.ones((int(N),1)))
    grad_b = np.sum(G, axis=1, keepdims=True)/ N
    
    print grad_b.shape
    assert(grad_W.shape == (K,d))
    assert(grad_b.shape == (K,1))
    
    return grad_W, grad_b


# In[70]:


grad_W,grad_b=ComputeGradients(X,Y,P,W,lambd)


# ## Gradient Checking

# In[71]:


def ComputeGradsNum(X, Y, W, b):
#function [grad_b, grad_W] = ComputeGradsNum(X, Y, W, b, lambda, h) 
    h=float(1e-6)
    grad_W = np.zeros_like(W)
    grad_b = np.zeros_like(b)
    
    P=EvaluateClassifier(X,W,b)
    c = ComputeCost_CrossEntropy(P,Y,W)
    
    for i in range(len(b)):
        b_try = b
        b_try[i] = b_try[i] + h
        P=EvaluateClassifier(X,W,b_try)
        c2 = ComputeCost_CrossEntropy(P,Y,W)
        grad_b[i] = (c2-c) / h
        
    for i,j in np.ndindex(W.shape):
        W_try = W
        W_try[i,j] = W_try[i,j] + h
        P=EvaluateClassifier(X,W_try,b)
        #change params for W and b
        c2 = ComputeCost_CrossEntropy(P,Y,W)
        grad_W[i,j] = (c2-c) / h
        
    return grad_W,grad_b


def gradient_check(X,Y,W,b, epsilon=1e-6):

    #type 1 numerical gradient
    Wplus = W + epsilon                               
    Wminus = W - epsilon  
    P=EvaluateClassifier(X,Wplus,b)
    J_plus = ComputeCost_CrossEntropy(P,Y,Wplus)
    P=EvaluateClassifier(X,Wminus,b)
    J_minus = ComputeCost_CrossEntropy(P,Y,Wminus)
    gradapprox = (J_plus - J_minus) / (2 * epsilon)  
   
    #type 2 numerical gradient
    gradWnum,gradbnum=ComputeGradsNum(X,Y,W,b)
    #gradapprox=grad_w_num
    grad,grad_b =ComputeGradients(X,Y,P,W,lambd)
    numerator = np.linalg.norm(grad - gradapprox)                      
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)    
    difference = numerator / denominator                               
    
    if difference < 1e-6:
        print("The gradient is correct!")
    else:
        print("The gradient is wrong!")
        print("Variation of :"+str(difference))
    
    return difference


# In[72]:


gradient_check(X,Y,W,b)


# In[ ]:


#Training
for i in range(10000):
        P=EvaluateClassifier(X,W,b)
        # Get the cost of the predictions
        costd=ComputeCost_CrossEntropy(P, Y, W)
        # Get accuracy of predictions
        acc=ComputeAccuracy(P,Y)
        # Learn the gradients of the cost function
        grad_W,grad_b=ComputeGradients(X,Y,P,W,lambd)
        # update params
        W,b=update_params(W,b,grad_W,grad_b,0.001)
        print('Step '+str(i)+": Cost:"+str(cost)+" Accuracy: "+str(acc)) 

