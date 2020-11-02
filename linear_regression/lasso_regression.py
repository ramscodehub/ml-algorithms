#(house pricing)hypothesis 1: w1*x + c  -----underfits the data(not even fitting the training data properly) high biased
#hypothesis 2: w1*x + w2*x^2 + c  ....fits the training data well
#hypothesis 3: w1*x + w2*x^2 + w2*x^3 + w2*x^4 +c  ...fits exactly the training data but cannot generailise the unseen data(overfit)high variance
#overfitting:If we have too many features the learned hypothesis may fit the training data very well(cost becomes approx 0)but fail to generailise the unseen data
#Regularisation:keep all the features but reduce the magnitude of parameters(works well when we have too many features with less training data)
# gennerailised model should have low bias and low variance
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
class LassoRegression:
    def __init__(self,learningRate,num_iter,penality):
        self.learningRate=learningRate
        self.num_iter=num_iter
        self.penality=penality
    def fit(self,X,Y):
        self.X=X
        self.Y=Y
        self.b=0
        self.m,self.n=X.shape
        self.W=np.zeros(self.n)
        for i in range(self.num_iter):
            self.updateWeights()
        return self
    #using gradient descent to update weights
    def updateWeights(self):
        YPredicted=self.predict(self.X)
        dW=np.zeros(self.n)
        for j in range(self.n):
            if self.W[j]>0:
                dW[j]=(-(2*(self.X.T).dot(self.Y-YPredicted))+(self.penality))/self.m
            else:
                dW[j] = (-(2*(self.X.T).dot(self.Y-YPredicted))-(self.penality))/self.m

        db=(-2*np.sum(self.Y-YPredicted))/self.m
        self.W=self.W-self.learningRate*dW
        self.b=self.b-self.learningRate*db
        return self
    def predict(self,X):
        return X.dot(self.W)+self.b

from sklearn.metrics import mean_squared_error
a=pd.read_csv("Salary_Data.csv",delimiter=",")
X=a.iloc[:,:-1].values
Y=a.iloc[:,1].values
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(X,Y,test_size=1/3)
model=LassoRegression(0.01,1000,500)
model.fit(xtrain,ytrain)
ypred=model.predict(xtest)
print(model.W,model.b)
plt.scatter(xtest, ytest, color='blue')
plt.plot(xtest, ypred, color='orange')
plt.title('Salary vs Experience')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
