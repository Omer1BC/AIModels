import numpy as np
import math
#Parser function for the data (x1,x2,Label)
def parse(train_path,test_path):
  train = []
  test = []
  with open(train_path) as f:
    for line in f:
      train.append(line.strip().split(","))
  with open(test_path) as f:
    for line in f:
      test.append(line.strip().split(","))
    return ([[[float(x),float(y)],int(z)] for x,y,z in train[1:]],[[[float(x),float(y)],int(z)] for x,y,z in test[1:]])
  
train_path = "train_data.txt"
test_path = "test_data.txt"
train,test = parse(train_path,test_path)

#Define the architecture of the network
L = 4
I = 2
H = 3
O = 1

#Activation function
def o(a):
  return 1/(1+np.exp(-a))
def deriv(a):
  return a*(1-a)
def init_W():
  w0 = np.ones((I,1),dtype=float)
  w1 = np.random.uniform(-10,10,(H,I))
  w2 = np.random.uniform(-10,10,(H,H))
  w3 = np.random.uniform(-10,10,(O,H))
  return np.array([w0,w1,w2,w3],dtype=object)

def predict(W,x):
  A = [x]
  for l in range(1,L):
    A.append(o(np.dot(W[l],A[-1])))
  A = np.array(A,dtype=object)
  return A
#Compute the error in a backwards recursive fashion
def backprop(W,W_GRAD,A,y,layer=1):
  if layer == L:
    #Final layer has an explicit cost
    return np.array([[2*(A[layer-1][0,0]-y)]]) *deriv(A[layer-1])
  else:
    #All other layers depend on the cost of the next layer
    delta = backprop(W,W_GRAD,A,y,layer+1)
    W_GRAD[layer] += np.dot(delta,A[layer-1].T)
    #Propagate the cost with respect to this layer's activation
    return np.dot(W[layer].T,delta) * deriv(A[layer-1])
epochs = 100
rate = .0001

best_s = 1
best_acc = .5
for s in [13]:
  np.random.seed(s)
  #Initialize the set of weights for each inter-layer boundary
  W = init_W()

  for e in range(epochs):
    W_GRAD = init_W() -1
    for x,y in train:
      #Forward pass
      A = predict(W,np.array(np.array(x).reshape(-1,1)))
      #Backward pass
      backprop(W,W_GRAD,A,y)
    for i in range(len(W)):
      #Update the weights following the gradient
      W[i] -= rate*(W_GRAD[i])

    count = 0
    #Model the accuracy
    for x,y in train:
      A = predict(W,np.array(x))
      if A[L-1][0] > .5 and y == 1:
        count += 1
      elif A[L-1][0] < .5 and y == 0:
        count += 1
    print(count/len(train),s)
    if (best_acc < count/len(train)):
      best_acc = count/len(train)
      best_s = s
print(best_s,best_acc)