# -*- coding: utf-8 -*-
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

class SVM():
    def __init__(
                 self, 
                 C, 
                 num_iterations, 
                 epsilon=0.001, 
                 kernel_function="linear", 
                 degree=3,
                 gamma=1,
                 bias=0
                ):
        
        self.C = C
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.kernel_function = kernel_function
        self.degree = degree
        self.gamma = gamma
        self.bias = bias
        
    
    def _satisfy_kkt(self, i):
        yg = self.y[i] * self._g(i)
        
        if abs(self.alpha[i]) < self.epsilon:
            return yg >= 1
        elif abs(self.alpha[i] - self.C) < self.epsilon:
            return yg <= 1
        else:
            return abs(yg - 1) < self.epsilon
        
    def _g(self, i):
        res = self.b
        
        for j in range(self.N):
            res += self.alpha[j] * self.y[j] * self.kernel(self.X[i], self.X[j])
            
        return res
    
    def _E(self, i):
        return self._g(i) - self.y[i]
    
    def kernel(self, xi, xj):
        if self.kernel_function == "linear":
            return sum(xi * xj)
        
        if self.kernel_function == "polynomial":
            return (self.gamma * sum(xi * xj) + self.bias) ** self.degree
        
        if self.kernel_function == "rbf":
            return np.exp(-self.gamma * sum((xi - xj) ** 2))
        
    def _select_alpha(self):
        inds = [i for i in range(self.N)] 
        
        inds1 = list(filter(lambda i: self.alpha[i] > 0 and self.alpha[i] < self.C, inds))
        inds2 = list(set(inds) - set(inds1))
        
        inds1.extend(inds2)
        
        for i in inds1:
            if self._satisfy_kkt(i):
                continue
            
            E1 = self.E[i]
            
            max_j = (0, 0)
            
            for j in inds:
                if i == j:
                    continue
                
                E2 = self.E[j]
                
                if abs(E1 - E2) > max_j[0]:
                    max_j = (abs(E1 - E2), j)

            return i, max_j[1]
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        
        self.b = 0.0
        
        self.N = len(y)
        
        self.feature_dim = X.shape[1]
        
        self.alpha = [0.0] * self.N
        
        self.E = [self._E(i) for i in range(self.N)]
        
        for i in range(self.num_iterations):
            try:
                a1, a2 = self._select_alpha() 
            except:
                break
            
            if self.y[a1] == self.y[a2]:
                L = max(0, self.alpha[a1] + self.alpha[a2] - self.C)
                H = min(self.C, self.alpha[a1] + self.alpha[a2])
                
            else:
                L = max(0, self.alpha[a2] - self.alpha[a1])
                H = min(self.C, self.C + self.alpha[a2] - self.alpha[a1])
                
            E1, E2 = self.E[a1], self.E[a2]
            
            K11 = self.kernel(self.X[a1], self.X[a1])
            K22 = self.kernel(self.X[a2], self.X[a2])
            K12 = self.kernel(self.X[a1], self.X[a2])
            eta = K11 + K22 - 2 * K12
            
            alpha2_new = self.alpha[a2] + self.y[a2] * (E1 - E2) / (eta + 1e-6)
            
            alpha2_new = np.clip(alpha2_new, L, H)
            
            alpha1_new = self.alpha[a1] + self.y[a1] * self.y[a2] * (self.alpha[a2] - alpha2_new)
            
            b1_new = -E1 - self.y[a1] * self.kernel(self.X[a1], self.X[a1]) * \
                     (alpha1_new - self.alpha[a1]) - self.y[a2] * \
                     self.kernel(self.X[a2], self.X[a1]) * \
                     (alpha2_new - self.alpha[a2]) + self.b
            
            
            b2_new = -E2 - self.y[a1] * self.kernel(self.X[a1], self.X[a2]) * \
                     (alpha1_new - self.alpha[a1]) - self.y[a2] * \
                     self.kernel(self.X[a2], self.X[a2]) * \
                     (alpha2_new - self.alpha[a2]) + self.b

                        
            if alpha1_new > 0 and alpha1_new < self.C:
                self.b = b1_new
                
            elif alpha2_new > 0 and alpha2_new < self.C:
                self.b = b2_new
                
            else:
                self.b = (b1_new + b2_new) / 2
            
            self.alpha[a1] = alpha1_new
            self.alpha[a2] = alpha2_new
            
            self.E[a1] = self._E(a1)
            self.E[a2] = self._E(a2)
            
    def predict(self, X):
        preds = []
        for x in X:
            pred = self.b
            for i in range(self.N):
                pred += self.alpha[i] * self.y[i] * self.kernel(x, self.X[i])
            if pred > 0:
                preds.append(1)
            else:
                preds.append(-1)
        preds = np.array(preds)
        return preds
        
    def score(self, X, y):
        preds = self.predict(X)
        score = accuracy_score(y, preds)
        return score
        
            
if __name__ == "__main__":
    raw_data = load_breast_cancer()
    
    X = raw_data.data
    y = raw_data.target
    y[y==0] = -1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=23323)
    
    svm = SVM(1000, 100, kernel_function="rbf", gamma=1)
    
    svm.fit(X_train[:100], y_train[:100])
    
    #print(svm.predict(X_train[:10]))
    #print(y_train[:0])
    print(svm.score(X_test[:100], y_test[:100]))