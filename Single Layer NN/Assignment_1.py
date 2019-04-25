#!/usr/bin/env python
# coding: utf-8

# ## DD2424 deep19 VT19-1 Deep Learning in Data Science - Assignment 1
# 
# ### Table of Content
# [Section 1](#Section_1) - Getting the data  
# [View Images](#view_images) - Scroll through the images (should run cell to see widget)   
# [Section 2](#Section_2) - Parameter Initialization  
# [Section 3](#Section_3) - Evaluate Classifier/ Forward Prop  
# [Section 4](#Section_4) - Compute the cost  
# [Section 5](#Section_5) - Compute Accuracy  
# [Section 6](#Section_6) - Compute and check the gradient descent  
# [Section 7](#Section_7) - Mini Batch Gradient Descent  
# [Trained Weights](#trained_weights) - Visualizing weights learnt  
# [Training_Runs](#Training_Runs) - Training with all 4 different settings  
# [Best Results](#bestPredictions) - Visulaization of best predictions on test data  
# [Comments on hyper-parameter tuning](#comments) - Comments on effects of lr and regularization
# 

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as Img
#from utils_assignment1 import LoadBatch, show_image, update_params
import pickle
import pprint
import seaborn as sns
from IPython.html.widgets import *
from tqdm import tqdm_notebook as tqdm
from scipy import misc
import sys

get_ipython().magic(u'matplotlib inline')
get_ipython().magic(u'matplotlib notebook')

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')

np.random.seed(400)


# ## Exercise 1: Training a multi-linear classifier
# 
# #### Section 1  
# <a id='Section_1'></a>
# 
# ## Getting the Data

# In[2]:


def LoadBatch(file):
    """
    Loads all the data from Cifar-10
    
    Argumets:
    file -- Path to main data directory
    
    Returns: 
    train_set_x, train_set_y, valid_set_x, 
    valid_set_y, test_set_x, test_set_y    -- Splits the dataset into 
    classes -- all the labels
    """
    #load data
    train_file = file+'/data_batch_1'
    valid_file = file+'/data_batch_2'
    test_file = file+'/test_batch'
    meta_file = file+'/batches.meta'
    
    with open(train_file, 'rb') as fo:
         train_set = pickle.load(fo)
    with open(valid_file, 'rb') as fo:
         valid_set = pickle.load(fo)
    with open(test_file, 'rb') as fo:
         test_set = pickle.load(fo)
    with open(test_file, 'rb') as fo:
         test_set = pickle.load(fo)
    with open(meta_file, 'rb') as fo:
         meta = pickle.load(fo)
    print("Load data " +u'\u2713' )

    #uncomment to get data overview
    #pprint.pprint(train_set)
    #for keys,value in train_set.items():
    #    print(keys)
    
    # data formatting to double
    train_set_x = np.array(train_set["data"][:],dtype='d') 
    train_set_y = np.array(train_set["labels"][:]) 
    valid_set_x = np.array(valid_set["data"][:],dtype='d') 
    valid_set_y = np.array(valid_set["labels"][:]) 
    test_set_x  = np.array(test_set["data"][:],dtype='d') 
    test_set_y  = np.array(test_set["labels"][:])
    classes=np.array(meta["label_names"][:])
    print("Image data to Double " +u'\u2713' )

    #standardising image data
    train_set_x = train_set_x/255.
    valid_set_x = valid_set_x/255.
    test_set_x = test_set_x/255.
    print("Standardizing image " +u'\u2713' )

    #one-hot encoding labels
    n_values = len(classes)
    train_set_y = np.eye(n_values)[train_set_y]
    valid_set_y = np.eye(n_values)[valid_set_y]
    test_set_y = np.eye(n_values)[test_set_y]
    print('one hot encoding '+ u'\u2713')
    
    return train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y, classes


# In[3]:


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


# ## Sneak peak  into the dataset 

# In[4]:


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
print ("classes shape :" + str(classes.shape))


# ## Check out some pics
# <a id='view_images'></a>

# In[67]:


def show_image(index):
    """
    Displays images from train_set_x i.e the raw set without dimetionality change
    
    Arguments -- index of the image from 1-10,000
    
    Returns -- None, Use slider to view all images
    """
    #for train_set_x
    sample = train_set_x[index]
    sample = sample*255.
    R = sample[0:1024].reshape(32,32)    
    G = sample[1024:2048].reshape(32,32)    
    B = sample[2048:].reshape(32,32)    
    img = np.dstack((R,G,B))
    img = Img.fromarray(img.astype("uint8"))
    plt.imshow(img,interpolation='bicubic')
    label = np.where(train_y[:,index]==1)
    plt.title(classes[label][0])


# In[68]:


interact(show_image,index = widgets.IntSlider(min = 0, max = 9999, step = 1, value=333))


# #### Section 2
# <a id='Section_2'></a>
# 
# ## Paramater Initialization

# In[7]:


def initialize_parameters(N, d, layers):
    """
    Initialize all the params need for the neural network
    
    Arguments:
    layers -- number of layers for the NN (here 1)
    N -- number of images NN is trained on
    d -- dimensions of input imagetaken from each sample, 
         could vary to check the gradients faster
         
    Returns:
    X -- Input to NN
    Y -- Ground Truth
    parameters -- NN parameters Ws and b(s)
    GDparams -- parameters to control the Gradient descent
    lambd -- regularization co-eff
    """
    
    #Number of possible predictions
    K = len(classes)
    #Subset of data to be trained on
    X = train_x[:,:N]
    Y = train_y[:,:N]
    #Gausssian parameters for W, b
    mu = 0
    sigma = 0.01

    parameters = {}
    L = layers           # number of layers in the network

    for l in range(1, L+1):
        parameters['W' + str(l)] = np.random.normal(loc = mu, scale = sigma, size = (K,d)) 
        parameters['b' + str(l)] = np.random.normal(loc = mu, scale = sigma, size = (K,1))
        assert(parameters['W' + str(l)].shape == (K, d))
        assert(parameters['b' + str(l)].shape == (K, 1))

    #regularization co-eff
    lambd = 0
    #n_batch=batch_size NOT the number of batches
    GDparams = {}
    GDparams['n_batch'] = 100
    GDparams['eta'] = 0.001
    GDparams['n_epochs'] = 200

    #Get the shapes right
    assert(X.shape == (d, N))
    assert(Y.shape == (K, N))
    assert(classes.size == 10)
    
    return X, Y, parameters, GDparams, lambd


# In[8]:


#Constants
N = 9
d = num_px
K = len(classes)
X, Y, parameters, GDparams, lambd = initialize_parameters(N, d, layers = 1)
plt.rcParams['figure.figsize'] = [8, 2]
plt.subplot(121).set_title("W")
ax = sns.heatmap(parameters['W1'])
plt.subplot(122).set_title("b")
ax = sns.heatmap(parameters['b1'])


# ####    Section 3 Evaluation
# <a id='Section_3'></a>
# 
# 
# ## Forward Propogation

# In[9]:


def softmax(x):
    """
    Instead of using np.exp(x)/np.sum(np.exp(x))
    I decrease the value of x with the max, 
    it could be any number as it cancels out when equation is expanded
    This is to avoid overflow due to exponential increase
    
    Argumets:
    x -- Product W, X
    
    Returns:
    soft -- softmax(W,X)
    """
    e_x = np.exp(x - np.max(x, axis = 0))
    soft = e_x / np.sum(e_x, axis = 0)
    return soft

def EvaluateClassifier(X, W, b):
    """
    s = W x + b
    p = SOFTMAX (s) 
    
    Arguments:
    X -- Input data
    W -- Weights
    b -- bias
    
    Returns:
    P -- Probabilities of each class for each sample in X
    
    """
    s = np.dot(W,X) + b
    P = softmax(s)
    #Dimensional check -- removed for minibatches
    #assert(X.shape == (d,N)),X.shape
    #assert(P.shape == (K,N))
    return P


# In[10]:


P = EvaluateClassifier(X,parameters["W1"],parameters["b1"])


# In[11]:


def visulaize_predictions(P, Y):
    """
    Just an excuse to use the beautiful seaborn :)
    
    Arguments:
    P -- Probabilties to get the predicitons
    Y -- Ground truth
    
    Returns:
    None -- Nice plots
    """
    
    plt.rcParams['figure.figsize'] = [25, 5]
    plt.subplot(141).set_title("Prediction")
    predictions = (P == P.max(axis = 0,keepdims = 1)).astype(float)
    ax = sns.heatmap(predictions)
    plt.subplot(142).set_title("Ground Truth")
    ax = sns.heatmap(Y)
    plt.subplot(143).set_title("Correct Matches")
    matches = (np.multiply(Y,predictions)).astype(float)
    ax = sns.heatmap(matches)
    plt.subplot(144).set_title("Differences")
    matches = Y + predictions
    ax = sns.heatmap(matches)
    plt.show(ax)


# In[12]:


visulaize_predictions(P,Y)


# #### Section 4
# <a id='Section_4'></a>
# 
# ## Computing the cost

# In[13]:


def ComputeCost_CrossEntropy(P, Y, W):
    """
    Loss of current state of prediction w.r.t the ground truth
    
    Arguments:
    P -- Probabilities of each class of each sample in input data
    Y -- Ground truth
    W -- Weights that produced the probabilities (Can be used in num. GD as well)

    Returns:
    cost -- sum of crossentropy loss and regularization term
    """
    
    P = np.reshape(P, (K, -1))
    ground_truth = np.reshape(Y, (K, -1))
    product = np.multiply(Y, P).sum(axis = 0)
    
    #cross entropy loss - Handling -log0 tending to infinity
    product[product == 0] = np.finfo(float).eps    #very low value
    crossEntropyLoss = np.mean(-np.log(product)) #.sum() / N

    
    L2_regularization_cost = lambd * np.power(W, 2).sum()
    
    cost = crossEntropyLoss + L2_regularization_cost
    return  cost


# In[14]:


cost = ComputeCost_CrossEntropy(P, Y, parameters["W1"])
print(cost)


# #### Section 5
# <a id='Section_5'></a>
# 
# ## Accuracy of the predictions

# In[15]:


def ComputeAccuracy(P, Y):
    """
    instead of using function acc = ComputeAccuracy(X, y, W, b)
    and calculating P again lets use it from the from the prev. output
    
    Arguments:
    P -- Probabilities of each class
    Y -- Ground Truth
    
    Returns:
    Accuracy -- Number of matches btw ground truth and predicions from probabilities
    """
    predictions = np.argmax(P,axis=0)
    groundtruth = np.argmax(Y,axis=0)
    matches = np.sum(predictions == groundtruth)
    total = len(predictions)
    accuracy = (matches/float(total))
    assert(P.shape == Y.shape)
    return accuracy


# In[16]:


accuracy = ComputeAccuracy(P, Y)
print ("Accuracy :" +  str(accuracy))


# #### Section 6
# <a id='Section_6'></a>
# 
# ## Computing the Gradients

# In[17]:


def ComputeGradients(X, Y, P, W, b, lambd):
    """ 
    G -- dJ/dZ - g for Batch
    Z -- WX
    grad_W1 -- dJ/dW
    grad_b -- dJ/db
    grad_L -- dJ/dL Regularization term
    grad_W -- dJ/dW + Regularization term

    Argumets:
    X -- input data
    Y -- Ground truth
    P -- current probabilities
    W -- Current weights that generated the probabilities
    b -- current bias
    lambd -- lambda - Regularization term (to differ from the keyword itself)
    
    Returns:
    grad_W -- analytical gradient of W
    grad_b -- analytical gradient of b
    """
    
    grad_W = np.zeros_like(W)
    grad_b = np.zeros_like(b)
    N = X.shape[1]

    for i in range(N):
        x = X[:,i].reshape(-1,1)
        y = Y[:,i].reshape(-1,1)
        p = P[:,i].reshape(-1,1)
        g = -(y - p)
        grad_b += g
        grad_W += np.outer(g,x)    # or simply (g.dot(x.T)
        
    grad_W /= N
    grad_b /= N
    grad_W += 2 * lambd * W    #Regularization term

    assert(grad_W.shape == (K,d))
    assert(grad_b.shape == (K,1))
    
    return grad_W, grad_b


# In[18]:


grad_W, grad_b = ComputeGradients(X, Y, P, parameters["W1"], parameters["b1"], lambd)
print (np.unique(grad_W))    #just to check if its not null
print (np.unique(grad_b))


# ## Gradient Checking

# In[19]:


def compute_grads_num(X, Y, W, b):
    """
    Numerical gradient descent using finite difference method. 
    
    Arguments:
    X -- Input dataset/ subset
    Y -- Corresponding Ground Truth
    W -- Current Weights
    b -- current bias
    
    Returns:
    grad_W -- Numerical Gradient of W  
    grad_b -- Numerical Gradient of b
    """
    N = X.shape[0]
    grad_W = np.zeros(W.shape)
    grad_b = np.zeros(b.shape)

    h = 1e-8    #very small number by which you want to vary the params
    P = EvaluateClassifier(X,W,b)
    c = ComputeCost_CrossEntropy(P,Y,W)

    for i in tqdm(range(W.shape[0]), desc = "Wights"):
        for j in range(W.shape[1]):
            W[i, j] += h    #change the weights and see if the cost reduces!
            P = EvaluateClassifier(X,W,b) #!!!!As ComputeCost_CrossEntropy takes P directly, update it
            c2 = ComputeCost_CrossEntropy(P,Y,W);
            W[i, j] -= h
            grad_W[i, j] = (c2-c) / h; #if cost decreases, value negative,
                                       #hence add this grad to W, ViceVersa

    for i in tqdm(range(b.shape[0]), desc = "Bias"):
        b[i] += h;    #change the bias and see if the cost reduces!
        P = EvaluateClassifier(X,W,b) #!!!!As ComputeCost_CrossEntropy takes P directly, update it
        c2 = ComputeCost_CrossEntropy(P,Y,W);
        b[i] -= h;
        grad_b[i] = (c2-c) / h;

   
    return grad_W, grad_b


# In[20]:


grad_W2,grad_b2 = compute_grads_num(X,Y,parameters["W1"],parameters["b1"])
print(np.unique(grad_W2))    #Sanity check -- just to know they are not all 0's
print(np.unique(grad_b2))


# In[21]:


def gradient_check(gradAnly, gradNum, epsilon):
    """
    Calculating realtive error between analytically and numerically computed gradient
    and checking if they are small. (smaller than epsilon)
    
    Order of the gradients does not matter
    
    Arguments:
    gradAnly -- Analytical gradient
    gradNum -- Numerical gradient
    epsilon -- very small value by which gradNum and gradAnly could differ
    
    Returns:
    relative_error -- relative error of gradAnly and gradNum
    """
    grad1 = gradAnly
    grad2 = gradNum
    difference = np.linalg.norm(grad1 - grad2)  #Could simply use np.abs(grad1 - grad2).sum()      
    summation = np.linalg.norm(grad1) + np.linalg.norm(grad2)  #varitaion - 2.0988034043225143e-08 only
    denominator = max(epsilon, summation)    #to avoid division by 0
    relative_error = difference / denominator                                                     
    
    if relative_error < 1e-6:
        print(u'\u2714  ' + "The gradient is correct!")
    else:
        print(u'\u2718  ' + "The gradient is wrong!")
    
    return relative_error


# In[22]:


gradient_check(grad_W, grad_W2, 1e-6)   #check grad_W
gradient_check(grad_b, grad_b2, 1e-6)   #check grad_b


# In[23]:


def update_params(W, b, grad_W, grad_b, eta):
    """
    Updating Weights and bias with the calculated gradients
    
    Arguments:
    W -- prev. Weights
    b -- prev. bias
    eta -- learning rate
    
    Returns:
    W -- updated Weights
    b -- updated bias
    """
    W = W - eta*grad_W
    b = b - eta*grad_b
    
    return W,b


# In[24]:


W_new, b_new = update_params(parameters["W1"], parameters["b1"], grad_W, grad_b, GDparams['eta'])
print(np.unique(W_new)) # should be set to params["W1"] and params["b1"] 


# #### Section 7
# <a id='Section_7'></a>
# ## Mini Batch Gradient Descent

# In[25]:


def get_mini_batches(X, Y, n_batch): 
    """
    Convert the whole dataset into smaller chunks
    
    Arguments:
    X -- Total input data
    Y -- Corresponding ground truth
    n_batch -- Number of samples in each batch (different from number of batches)
    
    Returns:
    mini_batches -- collection of small datasets each of size atmost n_batch
    """
    
    mini_batches = []
    X = X.T
    Y = Y.T
    data = np.hstack((X, Y)) 
    np.random.shuffle(data) 
    n_minibatches = data.shape[0] // n_batch 
    i = 0

    for i in range(n_minibatches): #as n_minibatch is floor of data.shape[0]/n_batch
        mini_batch = data[i * n_batch:(i + 1)*n_batch, :] 
        X_mini = mini_batch[:, :-10]
        Y_mini = mini_batch[:, 3072:]     #if needed .reshape((mini_batch[0].shape,10))
        mini_batches.append((X_mini, Y_mini)) 

    if data.shape[0] % n_batch != 0: 
        mini_batch = data[(i + 1)*n_batch:data.shape[0],:]
        X_mini = mini_batch[:, :-10] 
        Y_mini = mini_batch[:, 3072:]     #if needed .reshape((mini_batch[0].shape,10))
        mini_batches.append((X_mini, Y_mini)) 

    import math
    assert(len(mini_batches) == math.ceil(float(X.shape[0])/n_batch))
    return mini_batches  


# In[26]:


mini_batches = get_mini_batches(X, Y, GDparams['n_batch'])
print("Number of batches :", len(mini_batches))


# In[27]:


def MiniBatchGD(X, Y, X_val, Y_val, GDparams, W, b, lambd):
    """
    Perform Minibatch gradient descent shuffling at each iteration
    
    Argumets:
    X -- Total train data
    Y -- corresponding Ground truth
    X_val -- Total validation data
    Y_val -- corresponding Ground truth
    GDparams -- batch size, number of epochs, learning rates
    W -- initial Weights
    b -- initial Bias
    
    Returns:
    W -- trained Weights
    b -- trained bias
    """
    
    #Set up the canvas to do live plotting of the training info
    log = {"training loss":[], "validation loss":[],
           "training acc":[],"validation acc":[]} #using terms cost and loss interchangably
    fig = plt.figure(figsize = (8,3))      #must include %matplotlib notebook
    Loss = fig.add_subplot(121)
    Accuracy = fig.add_subplot(122)
 
    #plt.ion() #iteractive mode on
    plt.subplot(121).set_title("Loss")
    plt.xlabel('Epoches', axes = Loss)
    plt.ylabel('Loss', axes = Loss)
    plt.subplot(122).set_title("Accuracy")
    plt.xlabel('Epoches', axes = Accuracy)
    plt.ylabel('Accuracy', axes = Accuracy)
    fig.show() 
    fig.canvas.draw()
    
    step = 0 #to display each step accuracy if needed
    for i in tqdm(range((GDparams['n_epochs']))):     #progress bar
       
        mini_batches = get_mini_batches(X, Y, GDparams['n_batch']) #shuffled every iteration
        for batch in mini_batches:
            #do transpose again as we did earlier in mini_batch to stack'em up
            X_b = batch[0].T
            Y_b = batch[1].T           
            P = EvaluateClassifier(X_b, W, b)    # Get some predictions of the minibatch
            #no need to calculate for each mini batch
            #cost = ComputeCost_CrossEntropy(P, Y_b, W)    # Get the cost of these predictions
            #acc = ComputeAccuracy(P, Y_b)     # Get accuracy of predictions
            grad_W, grad_b = ComputeGradients(X_b, Y_b, P, W, b, lambd)  # Learn the mini batch
            W, b = update_params(W, b, grad_W, grad_b, GDparams['eta']) # Tune the weights and bias
            #print('Step ' + str(step) + ": Cost:" + str(cost) + " Accuracy: " + str(acc))
            step += 1
            
        #check performance of new W, b on training data    
        P_train = EvaluateClassifier(X, W, b) 
        log["training loss"].append(ComputeCost_CrossEntropy(P_train, Y, W))
        log["training acc"].append(ComputeAccuracy(P_train, Y))
    
        #check performance of new W, b on validation data    
        P_val = EvaluateClassifier(X_val, W, b)     
        log["validation loss"].append(ComputeCost_CrossEntropy(P_val, Y_val, W))
        log["validation acc"].append(ComputeAccuracy(P_val, Y_val))
        
        #display the log
        sns.set() #seaborn for beautiful plots
        #Loss.clear() #clear last epoch's plot
        #Accuracy.clear() 
        Loss.plot(log["training loss"], 'c', label = "training")
        Loss.plot(log["validation loss"], 'm', label = "validation")
        Accuracy.plot(log["training acc"], 'c', label = "training")
        Accuracy.plot(log["validation acc"], 'm', label = "validation")
        if(i == 0): #cant get it to work anyother way
            Loss.legend()
            Accuracy.legend()
        fig.canvas.draw()   # draw
        
    return W, b


# ## Testing Mini-Batch results

# In[34]:


trained_W, trained_b = MiniBatchGD(X, Y, valid_x, valid_y, GDparams, parameters["W1"], parameters["b1"], lambd)
P_test = EvaluateClassifier(test_x, trained_W, trained_b) 
print("Loss on the test dataset :", ComputeCost_CrossEntropy(P_test, test_y, trained_W))
print("Accuracy on the test datset :", ComputeAccuracy(P_test, test_y))


# ## Visualizing Weights
# <a id='trained_weights'></a>
# 

# In[35]:


def visualise_weights(W):
    for i in range(W.shape[0]):
        plt.subplot(2, 5, i+1)
        plt.axis("off")
        plt.title(classes[i])
        plt.imshow(misc.toimage(W[i].reshape((3, 32, 32)).transpose(1,2,0)), interpolation = 'gaussian')


# In[36]:


visualise_weights(trained_W)


#  ## Training Runs
#  <a id='Training_Runs'></a>
# 
#  ### Setting 1

# In[54]:


#Constants
N = 10000
d = num_px
K = len(classes)
X, Y, parameters, GDparams, lambd = initialize_parameters(N, d, layers = 1)
#Overriding the GDparams
GDparams = {}
GDparams['n_batch'] = 100
GDparams['eta'] = 0.1
GDparams['n_epochs'] = 40
lambd = 0
trained_W, trained_b = MiniBatchGD(X, Y, valid_x, valid_y, GDparams, parameters["W1"], parameters["b1"], lambd)


# In[55]:


P_test = EvaluateClassifier(test_x, trained_W, trained_b) 
print("Loss on the test dataset :",ComputeCost_CrossEntropy(P_test, test_y, trained_W))
print("Accuracy on the test datset :",ComputeAccuracy(P_test, test_y))
visualise_weights(trained_W)


# ### Setting 2 - Best

# In[64]:


#Constants
N = 10000
d = num_px
K = len(classes)
X, Y, parameters, GDparams, lambd = initialize_parameters(N, d, layers = 1)
#Overriding the GDparams
GDparams = {}
GDparams['n_batch'] = 100
GDparams['eta'] = 0.01
GDparams['n_epochs'] = 40
lambd = 0
trained_W, trained_b = MiniBatchGD(X, Y, valid_x, valid_y, GDparams, parameters["W1"], parameters["b1"], lambd)


# In[65]:


P_test = EvaluateClassifier(test_x, trained_W, trained_b) 
print("Loss on the test dataset :",ComputeCost_CrossEntropy(P_test, test_y, trained_W))
print("Accuracy on the test datset :",ComputeAccuracy(P_test, test_y))
visualise_weights(trained_W)


# ### To visualize False Positives and True Negatives of the best run
# <a id='bestPredictions'></a>

# In[66]:


visulaize_predictions(P_test,test_y)


# ### Setting 3

# In[59]:


#Constants
N = 10000
d = num_px
K = len(classes)
X, Y, parameters, GDparams, lambd = initialize_parameters(N, d, layers = 1)
#Overriding the GDparams
GDparams = {}
GDparams['n_batch'] = 100
GDparams['eta'] = 0.01
GDparams['n_epochs'] = 40
lambd = 0.1
trained_W, trained_b = MiniBatchGD(X, Y, valid_x, valid_y, GDparams, parameters["W1"], parameters["b1"], lambd)


# In[60]:


P_test = EvaluateClassifier(test_x, trained_W, trained_b) 
print("Loss on the test dataset :",ComputeCost_CrossEntropy(P_test, test_y, trained_W))
print("Accuracy on the test datset :",ComputeAccuracy(P_test, test_y))
visualise_weights(trained_W)


# ### Setting 4

# In[61]:


#Constants
N = 10000
d = num_px
K = len(classes)
X, Y, parameters, GDparams, lambd = initialize_parameters(N, d, layers = 1)
#Overriding the GDparams
GDparams = {}
GDparams['n_batch'] = 100
GDparams['eta'] = 0.01
GDparams['n_epochs'] = 40
lambd = 1
trained_W, trained_b = MiniBatchGD(X, Y, valid_x, valid_y, GDparams, parameters["W1"], parameters["b1"], lambd)


# In[62]:


P_test = EvaluateClassifier(test_x, trained_W, trained_b) 
print("Loss on the test dataset :",ComputeCost_CrossEntropy(P_test, test_y, trained_W))
print("Accuracy on the test datset :",ComputeAccuracy(P_test, test_y))
visualise_weights(trained_W)


# ### Comments on Hyper-parameter tuning 
# <a id='comments'></a>

# #### From the 4 training runs, the following traits are observed
# 
# #### Ranking on test data
# 1. 37% lambda=0, n epochs=40, n batch=100, eta=.01
# 2. 34% lambda=.1, n epochs=40, n batch=100, eta=.01
# 3. 25% lambda=0, n epochs=40, n batch=100, eta=.1
# 4. 19% lambda=1, n epochs=40, n batch=100, eta=.01
# <!-- -->
# 
# #### Insignts
# <!-- -->
# 
# ##### Learning Rate
# 1. The lower the learning rate, the better/smooth the training
# 2. As we can see the loss curve has too many ups and downs, risk of overshooting
# <!-- -->
# 
# ##### Regularization
# 1. When your model is very shallow its better not to do much regularization
# 2. As higher value of lamdba directly increases cost, there are many fluctuations and
#     can observe it is not able find right weights to decrease the loss
