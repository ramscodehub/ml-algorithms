#m is noOf_Samples
#n is noOf_features
#theta is (theta0,theta1,theta2,....thetaan)
#X is feature vector where x0 is always 1
#hypothesis function h(x1,x2,x3,....xn) = theta0 + theta1*x1 + theta2*x2 +....+ thetan*xn
#h(x)=[theta0 theta1 theta2...thetan][1 x1 x2 x3 ..xn]^Transpose
#alpha is learning rate
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
#returns the hypothesis values of target Variables
def hypothesis(theta,X,n):
    h=np.ones((X.shape[0],1))
    theta = theta.reshape(1, n+1)  # [theta0 theta1 theta2...thetan]
    for i in range(0,X.shape[0]):
        h[i]=float(np.matmul(theta,X[i]))
    h=h.reshape(X.shape[0])
    return h

def gradient_Descent(theta,alpha,noOf_Iterations,h,X,y,n):
    #lets have the cost as 1 for all iterations and update it during the training process
    cost=np.ones(noOf_Iterations)
    for i in range(0,noOf_Iterations):
        theta[0]=theta[0]-((alpha/X.shape[0])*sum(h-y))
        for j in range(1,n+1):
            theta[j]=theta[j]-((alpha/X.shape[0])*sum((h-y)*(X.transpose()[j])))
        h=hypothesis(theta,X,n)
        #cost=(1/2*n)*sum((predicted-actual)^2)
        cost[i]=(1/X.shape[0])*0.5*sum(np.square(h-y))
    theta=theta.reshape(1,n+1)
    return theta,cost

def linearRegression_Multivariable(X,y,alpha,noOf_Iterations):
    n=X.shape[1]
    one_column=np.ones((X.shape[0],1))
    X=np.concatenate((one_column,X),axis=1)
    theta=np.zeros(n+1)
    h=hypothesis(theta,X,n)
    theta,cost=gradient_Descent(theta,alpha,noOf_Iterations,h,X,y,n)
    return theta,cost

def Preprocess_Data(filename):
    data=np.loadtxt(filename,delimiter=",")
    X_train = data[:]
    X_train = np.delete(X_train, 2, axis=1)
    y_train = data[:,2]
    #lets put all ones and change it while iterating
    mean = np.ones(X_train.shape[1])
    std = np.ones(X_train.shape[1])
    for i in range(0, X_train.shape[1]):
        mean[i] = np.mean(X_train.transpose()[i])
        std[i] = np.std(X_train.transpose()[i])
        for j in range(0, X_train.shape[0]):
            X_train[j][i] = (X_train[j][i] - mean[i])/std[i]
    return X_train,y_train


X_train,y_train=Preprocess_Data("data2.txt")
theta, cost = linearRegression_Multivariable(X_train, y_train,0.01,50)
print(theta,cost)
plt.plot([i for i in range(len(cost))],cost)
plt.show()


"""data2.txt
2104,3,399900
1600,3,329900
2400,3,369000
1416,2,232000
3000,4,539900
1985,4,299900
1534,3,314900
1427,3,198999
1380,3,212000
1494,3,242500
1940,4,239999
2000,3,347000
1890,3,329999
4478,5,699900
1268,3,259900
2300,4,449900
1320,2,299900
1236,3,199900
2609,4,499998
3031,4,599000
1767,3,252900
1888,2,255000
1604,3,242900
1962,4,259900
3890,3,573900
1100,3,249900
1458,3,464500
2526,3,469000
2200,3,475000
2637,3,299900
1839,2,349900
1000,1,169900
2040,4,314900
3137,3,579900
1811,4,285900
1437,3,249900
1239,3,229900
2132,4,345000
4215,4,549000
2162,4,287000
1664,2,368500
2238,3,329900
2567,4,314000
1200,3,299000
852,2,179900
1852,4,299900
1203,3,239500"""
