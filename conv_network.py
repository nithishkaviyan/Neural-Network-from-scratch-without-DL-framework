
# coding: utf-8

# In[189]:


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

num_outputs=10

def convolution(x_s,w):
    t=x_s*w
    sm=np.sum(t)
    return sm

def relu(z1):
    return z1*(z1>0)

def drelu(z):
    return 1*(z>0)

def softmax_function(z):
    ZZ = np.exp(z)/np.sum(np.exp(z))
    return ZZ

##One hot vector encoding
def one_hot(y):
    y_tr=np.zeros((1,10))
    y_tr[0,y]=1
    return(y_tr.T)

num_epochs=15

##Filter size
f=5
##Number of channels
n_c=4
x_tr=np.reshape(x_train,(-1,28,28))
m=60000

nh=x_tr.shape[1]
nw=x_tr.shape[2]

height=nh-f+1
width=nw-f+1

W=np.random.randn(f,f,n_c)/np.sqrt(f)
W2=np.random.randn(num_outputs,(height*width*n_c))/np.sqrt(num_outputs)
b2=np.zeros((num_outputs,1))


tot_correct=0
LR = .01
time1=time.time()
for epochs in range (num_epochs):
    #Learning rate schedule
    if (epochs > 5):
        LR = 0.001
    if (epochs > 10):
        LR = 0.0001
    if (epochs > 15):
        LR = 0.00001
        
    tot_correct=0
    for i in range(m):
        ##Convolutional Layer
        Z=np.zeros((height,width,n_c))
        k=randint(0,59999)
        x=x_tr[k]
        y=y_train[k]
        for h in range(height):
            for w in range(width):
                for c in range(n_c):
                    vs=h
                    ve=h+f
                    hs=w
                    he=w+f
                    x_slice=x[vs:ve,hs:he]
                    Z[h,w,c]=convolution(x_slice,W[:,:,c])
        A=relu(Z)
        ##Fully connected layer
        A1=np.reshape(A,(-1,1))

        Z2=np.matmul(W2,A1)+b2
        A2=softmax_function(Z2)
    
        p=np.argmax(A2)
        if(p==y):
            tot_correct+=1
    
        ##Backprop FC
        y1=one_hot(y)
        dZ2 = y1-A2
        dW2=np.dot(dZ2,A1.T)
        db2=np.sum(dZ2,axis=1,keepdims=True)
        delta=np.matmul(W2.T,dZ2)
    
    
        ##Backprop Convolutional Layer
        delta_reshape=np.reshape(delta,(height,width,-1))
        dZ=delta_reshape*drelu(Z)

        dW=np.zeros((f,f,n_c))
        for h1 in range(W.shape[0]):
            for w1 in range(W.shape[1]):
                for c1 in range(W.shape[2]):
                    vs1=h1
                    ve1=h1+f
                    hs1=w1
                    he1=w1+f
                    x_slice1=x[vs1:ve1,hs1:he1]
                    dW[:,:,c1] += x_slice1*dZ[h1,w1,c1]
    
        ##Update Step
        W += LR*dW
        W2 += LR*dW2
        b2 += LR*db2
    
    print("Epoch: "+str(epochs+1)+" Train Accuracy: "+str(tot_correct/m)) 
    
time2=time.time()
print("Running Time: "+str(time2-time1))

#Test data
total_correct = 0
for n in range( len(x_test)):
    y = y_test[n]
    x = np.reshape(x_test[n][:],(28,28))
    Z=np.zeros((height,width,n_c))
    for h in range(height):
        for w in range(width):
            for c in range(n_c):
                vs=h
                ve=h+f
                hs=w
                he=w+f
                x_slice=x[vs:ve,hs:he]
                Z[h,w,c]=convolution(x_slice,W[:,:,c])
    A=relu(Z)
    ##Fully connected layer
    A1=np.reshape(A,(-1,1))
    Z2=np.matmul(W2,A1)+b2
    A2=softmax_function(Z2)
    p=np.argmax(A2)
    if(p==y):
        total_correct+=1
print("Test Accuracy :"+ str(total_correct/np.float(len(x_test) ) ))

