import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
class ElasticNetRegression:
    def __init__(self, learningRate, num_iter, penality1,penality2):
        self.learningRate = learningRate
        self.num_iter = num_iter
        self.penality1 = penality1
        self.penality2=penality2

    def fit(self, X, Y):
        self.X = X
        self.Y = Y
        self.b = 0
        self.m, self.n = X.shape
        self.W = np.zeros(self.n)
        for i in range(self.num_iter):
            self.updateWeights()
        return self
    #using gradient descent to update weights

    def updateWeights(self):
        YPredicted = self.predict(self.X)
        dW=np.zeros(self.n)
        for j in range(self.n):
            if self.W[j]>0:
                dW[j] = (-(2*(self.X.T).dot(self.Y-YPredicted))+self.penality1+2*self.penality2*self.W)/self.m
            else:
                dW[j] = (-(2*(self.X.T).dot(self.Y-YPredicted))-self.penality1+2*self.penality2*self.W)/self.m
       
        db = (-2*np.sum(self.Y-YPredicted))/self.m
        self.W = self.W-self.learningRate*dW
        self.b = self.b-self.learningRate*db
        return self

    def predict(self, X):
        return X.dot(self.W)+self.b


