import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import MinMaxScaler


class LogisticRegression:
    def __init__(self, lr, epochs, num_of_features) -> None:
        self.num_of_features = num_of_features
        self.lr = lr
        self.epochs = epochs
        self.num_of_features = num_of_features
        self.weights = np.zeros((self.num_of_features, 1))
        self.bias = 0

    def normalize(self, X):
        m, n = X.shape
        
        # Normalizing all the n features of X.
        for i in range(n):
            X = (X - X.mean(axis=0))/X.std(axis=0)
            
        return X

    def sigmoid(self, z):
        return 1.0/(1+np.exp(-z))
    
    def predict(self, X):
        return self.sigmoid(np.dot(X, self.weights) + self.bias)
    
    def loss(self, y, y_hat):
        loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))
        return loss
    
    def gradients(self, X, y, y_hat):
        # m-> number of training examples.
        m = X.shape[0]
        
        # Gradient of loss w.r.t weights.
        dw = (1/m)*np.dot(X.T, (y_hat - y))
        
        # Gradient of loss w.r.t bias.
        db = (1/m)*np.sum((y_hat - y)) 
        
        return dw, db

    def fit(self, X, y):
        m = X.shape[0]
        # Reshaping y.
        y = y.reshape(m,1)

        # Empty list to store losses.
        losses = []
        
        # Training loop.
        for _ in range(self.epochs):     
            y_hat = self.predict(X)
            dw, db = self.gradients(X, y, y_hat)
            
            # Updating the parameters.
            self.weights -= self.lr*dw
            self.bias -= self.lr*db
            
            loss = self.loss(y, y_hat)
            losses.append(loss)

        # returning weights, bias and losses(List).
        return self.weights, self.bias

    def plot(self, X, y):
        fig = plt.figure(figsize=(10,8))
        plt.plot(X[:, 0][y==0], X[:, 1][y==0], "g^")
        plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs")

        plt.xlabel("feature 1")
        plt.ylabel("feature 2")
        plt.title('Decision Boundary')
        x1 = [min(X[:,0]), max(X[:,0])]
        m = -self.weights[0]/self.weights[1]
        c = -self.bias/self.weights[1]
        x2 = m*x1 + c
        plt.plot(x1, x2, 'y-')
        plt.show()


X, y = make_classification(n_features=2, n_redundant=0, 
                           n_informative=2, random_state=1, 
                           n_clusters_per_class=1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

model = LogisticRegression(lr=1e-2, epochs=2000, num_of_features=2)
model.fit(X, y)
model.plot(X, y)

