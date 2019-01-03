
# coding: utf-8

# In[2]:


import numpy as np
import h5py
import time
import copy
from random import randint

#load MNIST data
MNIST_data = h5py.File('MNISTdata.hdf5', 'r')
x_train = np.float32(MNIST_data['x_train'][:] )
y_train = np.int32(np.array(MNIST_data['y_train'][:,0]))
x_test = np.float32( MNIST_data['x_test'][:] )
y_test = np.int32( np.array( MNIST_data['y_test'][:,0] ) )
MNIST_data.close()

#Implementation of stochastic gradient descent algorithm
#number of inputs
num_inputs = 28*28
#number of units in hidden layer
num_hidden = 100
#number of outputs
num_outputs = 10

model = {}
model['W1'] = np.random.randn(num_hidden,num_inputs) / np.sqrt(num_inputs)
model['W2'] = np.random.randn(num_outputs,num_hidden) / np.sqrt(num_hidden)
model['b1'] = np.zeros((num_hidden,1))
model['b2'] = np.zeros((num_outputs,1))

model_grads = copy.deepcopy(model)

def relu(z):
    return z*(z>0)

def drelu(z):
    return 1*(z>0)

def sigmoid(k):
    k1=1/(1+np.exp(-k))
    return k1

def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ

##One hot vector encoding
def one_hot(y):
    y_tr=np.zeros((1,10))
    y_tr[0,y]=1
    return(y_tr.T)

def forward(x,y, model):
    in_model={}
    Z1 = np.dot(model['W1'], x)+model['b1']
    #a1 = sigmoid(Z1)
    a1=relu(Z1)
    Z2 = np.dot(model['W2'],a1)+model['b2']
    in_model={'Z1':Z1,'a1':a1,'Z2':Z2}
    p = softmax_function(Z2)
    return (p,in_model)

def backward(x,y,p, model, model_grads,in_model):
    y1=one_hot(y)
    dZ2 = y1-p
    
    a1=in_model['a1']
    
    model_grads['W2']=np.dot(dZ2,a1.T)
    model_grads['b2']=np.sum(dZ2,axis=1,keepdims=True)
    
    da1=np.dot(model['W2'].T,dZ2)
    #dZ1=da1*(a1*(1-a1))
    dZ1=da1*drelu(a1)
    model_grads['W1']=np.dot(dZ1,np.transpose(x))
    model_grads['b1']=np.sum(dZ1,axis=1,keepdims=True)
    
    return model_grads

import time
time1 = time.time()
LR = .01
num_epochs =20
for epochs in range(num_epochs):
    #Learning rate schedule
    if (epochs > 5):
        LR = 0.001
    if (epochs > 10):
        LR = 0.0001
    if (epochs > 15):
        LR = 0.00001
    total_correct = 0
    for n in range( len(x_train)):
        n_random = randint(0,len(x_train)-1 )
        y = y_train[n_random]
        x = np.reshape(x_train[n_random][:],(1,-1))
        p, inter_model = forward(x.T, y, model)
        prediction = np.argmax(p)
        if (prediction == y):
            total_correct += 1
        model_grads = backward(x.T,y,p, model, model_grads,inter_model)
        model['W1'] = model['W1'] + LR*model_grads['W1']
        model['b1'] = model['b1'] + LR*model_grads['b1']
        model['W2'] = model['W2'] + LR*model_grads['W2']
        model['b2'] = model['b2'] + LR*model_grads['b2']
    print("Epoch: "+str(epochs+1)+"  Train Accuracy : " + str(total_correct/np.float(len(x_train) ) ))

time2 = time.time()
print(time2-time1)
######################################################
#test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = np.reshape(x_test[n][:],(1,-1))
    p,_ = forward(x.T, y, model)
    prediction = np.argmax(p)
    if (prediction == y):
        total_correct += 1
print("Test Accuracy :"+ str(total_correct/np.float(len(x_test) ) ))

