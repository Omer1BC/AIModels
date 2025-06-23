import json
import numpy as np
import matplotlib.pyplot as plt
import math as Math

def load_data(file):
  with open(file,"r") as file:
    data = json.load(file)
    train_x = data['X_train']
    train_y = data['Y_train']
    val_x = data['X_val']
    val_y = data['Y_val']
    test_x = data['X_test']
    test_y = data['Y_test']

  return[x for x in train_x],train_y,[x for x in val_x],val_y,test_x,test_y
def dot(u,v):
  c = 0
  for x,y in zip(u,v):
    c += x*y
  return c
def x_terms(x,n):
  return [x**i for i in range(n)]
class Model:
  def __init__(self,m=12,lam=0,eta=0,n=2):
    self.n = n
    self.w = [0]*self.n
    self.m = m
    self.lam = lam
    self.eta =eta
    # print('eta is ',self.eta)

  def update_wts(self,grad):
    # print(f'BEFORE UPDATE {grad} {self.eta}')
    for i in range(len(self.w)):
      self.w[i] -= self.eta*grad[i]
    # print(self.w,grad)
  def h(self,x):
    c= 0
    for u,v in zip(self.w,x):
      c += u*v
    return c


  def resize(self,X):
    return [x_terms(x,self.n) for x in X]

  def train(self, X,Y):
    cost = 0
    X_VEC = self.resize(X)
    for i in range(0,len(X),self.m):
      grad = [0]*self.n
      # print(f'{i} {self.m} {X_VEC} | {type(i+self.m)} {type(len(X_VEC))}')
      for j in range(i,min(i+self.m,len(X_VEC))):
        # print(f' {j}')
        x = X_VEC[j]
        y = Y[j]
        h = self.h(x)
        # print(f' x: {x} y: {y} w: {self.w} h: {h}')
        soln = []
        for k in range(self.n):
          cost += (1/(2*self.m))*(h-y)**2
          term = (1/self.m)*(h-y)*x[k] if k > 0 else (1/self.m)*(h-y)
          soln.append(term)
          grad[k] += term
        # print(f'    current gradient terms to be added {soln}')

      soln = []
      for k in range(self.n):
        cost += (self.lam/(2*self.m)) * (self.w[k]**2) if k > 0 else 0
        term2 = (self.lam/self.m)*self.w[k] if k > 0 else 0
        grad[k] += term2
        soln.append(term2)

      # print(f'    normalization terms to be added {soln} | {grad}')
      self.update_wts(grad)
      return cost
  def validate(self,X,Y):
    X_VEC = self.resize(X)
    cost = 0
    for i in range(0,len(X),self.m):
      for j in range(i,min(i+self.m,len(X_VEC))):
        x = X_VEC[j]
        y = Y[j]
        h = self.h(x)
        soln = []
        for k in range(self.n):
          cost += (1/(2*self.m))*(h-y)**2
      soln = []
      for k in range((self.n)):
        cost += (self.lam/(2*self.m)) * (self.w[k]**2) if k > 0 else 0
      return cost




x,y, x_val,y_val,x_test,y_test = load_data("Data.json")

X_COR = x + x_val + x_test
Y_COR = y + y_val + y_test
plt.scatter(X_COR,Y_COR)
plt.title(f'Part: (a) - Scatter Plot')
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

lam = 1
n= 2
eta = 9.0e-7
lam_1_model = Model(len(x),lam=lam,eta=eta,n=n)
lam_1_model.train(x,y)
loss = lam_1_model.validate(x_test,y_test)

print(f'Cost with lambda = {lam} is {loss}')
lam = 0
lam_0_model = Model(len(x),lam=lam,eta=eta,n=n)
lam_0_model.train(x,y)



line_x = np.linspace(min(X_COR),max(X_COR),100)

X_COR = x
Y_COR = y
plt.scatter(X_COR,Y_COR)
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f'Part: (b) iv - Model Using Lambda = {0}')
line_y = dot(lam_0_model.w,x_terms(line_x,n))
plt.plot(line_x,line_y,label=f'Theta: {lam_0_model.w}')
plt.legend()
plt.show()

train_error = lam_0_model.validate(x_val,y_val)
validation_error = lam_0_model.validate(x_test,y_test)
plt.xlabel("X\n"+ f'Training Cost: {train_error} Validation Cost: {validation_error}')
plt.ylabel("Y")
plt.title(f' Model Using Lambda = {0}')
line_y = dot(lam_0_model.w,x_terms(line_x,n))
plt.plot(line_x,line_y,label=f'Theta: {lam_0_model.w}')
plt.ylim(min(Y_COR),max(Y_COR))
plt.legend()
plt.show()

DEGREE = [2,3,4]
ETA = [5.0e-13,3.0e-19,1.70e-25]
POLY_MODELS = { k: Model(lam=1,eta=ETA[i],n=k+1) for i,k in enumerate(DEGREE)}

plt.title(f' New Curves')
for k in POLY_MODELS:
  model = POLY_MODELS[k]
  model.train(x,y)
  train_error = model.validate(x_val,y_val)
  validation_error = model.validate(x_test,y_test)
  plt.xlabel("X")
  plt.ylabel("Y")
  line_y = dot(model.w,x_terms(line_x,model.n))
  plt.plot(line_x,line_y,label=f'Degree: {k}')
X_COR = x + x_val + x_test
Y_COR = y + y_val + y_test
plt.ylim(min(Y_COR),max(Y_COR))
plt.legend()
plt.show()

for k in POLY_MODELS:
  model = POLY_MODELS[k]
  model.train(x,y)
  train_error = model.validate(x_val,y_val)
  validation_error = model.validate(x_test,y_test)
  plt.xlabel("X\n"+ f'Training Cost: {train_error} Validation Cost: {validation_error}')
  plt.ylabel("Y")
  line_y = dot(model.w,x_terms(line_x,model.n))
  plt.plot(line_x,line_y,label=f'Degree: {k}')
  plt.legend()
  plt.show()
#Part (d) v
target_model = POLY_MODELS[4]
lambda_accs = []
lambda_values =  [10**i for i in range(0,25)]
best_lam = 10**32
best_error = float('inf')
for lam in lambda_values:
  target_model.lam = lam
  target_model.train(x,y)

  error = target_model.validate(x_test,y_test)
  lambda_accs.append(error)
  if error < best_error:
    best_error = error
    best_lam = lam

plt.title(f' Learning Curve')
plt.plot(lambda_values,lambda_accs)
plt.xlabel("Lambda\nBest Lambda: " + f'{best_lam}')
plt.ylabel("Error")
plt.show()

plt.title(f' Fitting Curve with Best Lambda = {best_lam} and Best Test Error {best_error}')
line_y = dot(target_model.w,x_terms(line_x,target_model.n))

plt.plot(line_x,line_y)
plt.xlabel("X")
plt.ylabel("Y")
plt.show()