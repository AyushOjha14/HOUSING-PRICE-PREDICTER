import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = np.genfromtxt('ex1data2.txt',delimiter=',')
dim=data.shape
m=dim[0]
n=dim[1]-1
X=data[:,0:n]
y=data[:,n]
y=y[:,np.newaxis]
fig = plt.figure(figsize = (10, 7))
ax = plt.axes(projection ="3d")
ax.scatter3D(X[:,0],X[:,1],y, color = "green")
plt.title("HOUSING PRICE PREDICTION")
ones=np.ones((m,1))
X = np.hstack((ones, X))
initial_theta=np.zeros((n+1,1))
def costfunction(X,y,theta):
    m=y.shape[0]
    h=np.matmul(X,theta)
    d=np.subtract(h,y)
    e=d**2
    sm=np.sum(e)
    J=np.divide(sm,2*m)
    return J
def gradientdecent(X,y,theta,alpha,iterations):
    i=1
    m=y.shape[0]
    while i<=iterations:
        h=np.dot(X,theta)
        d=np.subtract(h,y)
        a=np.dot(X.transpose(),d)
        print(a)
        theta=theta-(alpha/m)*a
        i=i+1
    return theta
theta=gradientdecent(X,y,initial_theta,0.01,1400)
print(theta)
inp=input("ENTER THE FEATURES SEPARATED BY COMMA\n")
inp=inp.split(",")
inp=[float(i) for i in inp]
b=np.array(inp)
b=b[:,np.newaxis]
b=np.hstack((np.ones((1,1)), b))
prediction=np.matmul(b,theta)
plt.show()
print(prediction)

        
        
    


    
    
    
