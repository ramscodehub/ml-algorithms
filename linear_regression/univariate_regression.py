import numpy as np
from matplotlib import pyplot as plt
#applying gradient descent algorithm to optimise a mean square error for a single variable regression
#hypothesis: y=mx+c(linear regression in single variable)
#h(x)=x0+a1x1+a2x2+......anxn(hypothesis for multivariate linear regression)
def gradient_descent(x,y):
    mCurrent=0
    bCurrent=0
    noOfIterations=500
    learningRate=0.08
    n=len(x)
    c=[]
    for i in range(noOfIterations):
        yPredicted=(mCurrent*x)+bCurrent
        cost = (1/n)*sum([val*val for val in (y-yPredicted)])
        md=(-1/n)*sum(x*(y-yPredicted))
        bd=(-1/n)*sum(y-yPredicted)
        mCurrent=mCurrent-(learningRate*md)
        bCurrent=bCurrent-(learningRate*bd)
        c.append(cost)
    print(mCurrent,bCurrent,cost,i)
    
    plt.plot([i for i in range(0,noOfIterations)],c)
    plt.show()

x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])
gradient_descent(x,y)
