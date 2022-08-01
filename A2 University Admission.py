import numpy as np

data= np.genfromtxt('ex2data1.txt',delimiter=',')

dim=data.shape
m=dim[0]
n=dim[1]-1
X=data[:,0:n]
y=data[:,n]
y=y[:,np.newaxis]
ones=np.ones((m,1))

mean=np.mean(X,axis=0)
std=np.std(X,axis=0)
X= (X - mean)/std

X = np.hstack((ones, X))
initial_theta=np.zeros((n+1,1))

def sigmoid(z):
    return 1/ (1 + np.exp(-z))

def costFunction(theta, X, y):
    h = sigmoid(np.dot(X,theta))
    error = (-y * np.log(h)) - ((1-y)*np.log(1-h))
    cost = (1/m) * sum(error)
    grad = 1/m * np.dot(X.transpose(),(h - y))
    return cost, grad

cost, grad= costFunction(initial_theta,X,y)
def gradientDescent(X,y,theta,alpha,num_iters):
    m=len(y)
    for i in range(num_iters):
        cost, grad = costFunction(theta,X,y)
        theta = theta - (alpha * grad)
    return theta

theta=gradientDescent(X,y,initial_theta,0.1,10000)
print("THETA= ",theta)

def accu(theta,X):
    h = X.dot(theta)
    return h>0

p=accu(theta,X)
sm=sum(y)
sn=100-sm
print("Accuracy:", sum(p==y)[0],"%")
s=sum(p-y==-1)
print("ACCURACY OF ONES",(sm[0]-s[0])/sm[0]*100,"%")
s=sum(p-y==1)
print("ACCURACY OF ZEROS",(sn[0]-s[0])/sn[0]*100,"%")
    
