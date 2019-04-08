import pickle
import pprint
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def LoadBatch(file):
    
    #load data
    train_file=file+'/data_batch_1'
    valid_file=file+'/data_batch_2'
    test_file=file+'/test_batch'
    meta_file=file+'/batches.meta'
    with open(train_file, 'rb') as fo:
         train_set= pickle.load(fo)
    with open(valid_file, 'rb') as fo:
         valid_set= pickle.load(fo)
    with open(test_file, 'rb') as fo:
         test_set= pickle.load(fo)
    with open(test_file, 'rb') as fo:
         test_set= pickle.load(fo)
    with open(meta_file, 'rb') as fo:
         meta= pickle.load(fo)
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
    train_set_x=train_set_x/255.
    valid_set_x=valid_set_x/255.
    test_set_x=test_set_x/255.
    print("Standardizing image " +u'\u2713' )

    #one-hot encoding labels
    n_values = len(classes)
    train_set_y=np.eye(n_values)[train_set_y]
    valid_set_y=np.eye(n_values)[valid_set_y]
    test_set_y=np.eye(n_values)[test_set_y]
    print('one hot encoding '+ u'\u2713')
    
    return train_set_x, train_set_y, valid_set_x, valid_set_y, test_set_x, test_set_y, classes


def show_image(data,index):
    sample=data[index]
    sample=sample*255.
    R = sample[0:1024].reshape(32,32)    
    G = sample[1024:2048].reshape(32,32)    
    B = sample[2048:].reshape(32,32)    
    img = np.dstack((R,G,B))
    img = Image.fromarray(img.astype("uint8"))
    plt.imshow(img,interpolation='bicubic')













def update_params(W,b,grad_W,grad_b,eta):
    W=W-eta*grad_W
    b=b-eta*grad_b
    return W,b

def get_mini_batches(X, Y, n_batch): 
    mini_batches = []
    X=X.T
    Y=Y.T
    data = np.hstack((X, Y)) 
    np.random.shuffle(data) 
    n_minibatches = data.shape[0] // n_batch 
    i = 0

    for i in range(n_minibatches - 1): 
        mini_batch = data[i * n_batch:(i + 1)*n_batch, :] 
        X_mini = mini_batch[:, :-10] 
        Y_mini = mini_batch[:, 3072:].reshape((100,10))
        mini_batches.append((X_mini, Y_mini)) 

    if data.shape[0] % n_batch != 0: 
        mini_batch = data[i * n_batch:data.shape[0]] 
        X_mini = mini_batch[:, :-10] 
        Y_mini = mini_batch[:, 3072:].reshape((100,10))
        mini_batches.append((X_mini, Y_mini)) 

    return mini_batches 

def MiniBatchGD(X,Y,GDparams,W,b,lambd):
    step=0
    for i in range(GDparams['n_epochs']):
        mini_batches=get_mini_batches(X, Y, GDparams['n_batch'])
        #print ("Processing mini batches " + u'\u2713')
        for batch in mini_batches:
            #do transpose again 
            X=batch[0].T
            Y=batch[1].T
            # Get some predictions
            P=EvaluateClassifier(X,W,b)
            # Get the cost of the predictions
            cost=ComputeCost_CrossEntropy(X, Y, W,b,lambd)
            # Get accuracy of predictions
            acc=ComputeAccuracy(X,Y,W,b)
            # Learn the gradients of the cost function
            grad_W,grad_b=ComputeGradients(X,Y,P,W,lambd)
            # update params
            W,b=update_params(W,b,grad_W,grad_b,GDparams['eta'])
            print('Step '+str(step)+": Cost:"+str(cost)+" Accuracy: "+str(acc))
            step=step+1
        print("Epoch "+str(i+1)+ u'\u2713')
        

def grad_comparision(grad_W,grad_b,grad_W_num,grad_b_num):
    difference_w=np.sum(np.abs(grad_W-grad_W_num))
    sum_w=np.sum(np.abs(grad_W)+np.abs(grad_W_num))
    relative_w=difference_w/max(sum_w,float(1e-6))
    print ('W Diff: '+str(difference_w))
    print ('W Sum: '+str(sum_w))
    print ('W relative error: '+str(relative_w))
    
    difference_b=np.sum(np.abs(grad_b-grad_b_num))
    sum_b=np.sum(np.abs(grad_b)+np.abs(grad_b_num))
    relative_b=difference_b/max(sum_b,float(1e-6))
    print ('b Diff: '+str(difference_b))
    print ('b Sum: '+str(sum_b))
    print ('b relative error: '+str(relative_b))        
        
def Train_whole(X,Y,GDparams,W,b,lambd):
    for i in range(100):
        P=EvaluateClassifier(X,W,b)
        # Get the cost of the predictions
        cost=ComputeCost_CrossEntropy(X, Y, W,b,lambd)
        # Get accuracy of predictions
        acc=ComputeAccuracy(X,Y,W,b)
        # Learn the gradients of the cost function
        grad_W,grad_b=ComputeGradients(X,Y,P,W,lambd)
        # update params
        W,b=update_params(W,b,grad_W,grad_b,0.001)
        print('Step '+str(i)+": Cost:"+str(cost)+" Accuracy: "+str(acc)) 