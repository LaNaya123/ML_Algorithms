# -*- coding: utf-8 -*-
import numpy as np
import math
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

def calculate_entropy(x):
    labels = set([x[i] for i in range(len(x))])
    
    entropy = 0.0
    
    for label in labels:
        p = float(len(x[x == label])) / len(x)
        log_p = np.log2(p)
        entropy -= p * log_p
    return entropy

def calculate_condition_entropy(x, conditions):
    feature_values = set(conditions[i] for i in range(len(conditions)))
    
    condition_entropy = 0.0
    C = 0.0
    
    for feature_value in feature_values:
       sub_x = x[conditions == feature_value]
       entropy = calculate_entropy(sub_x)
       condition_entropy += (len(sub_x) / len(x)) * entropy
       
       p = len(sub_x) / len(x)
       log_p = np.log2(p)
       C -= p * log_p
    return condition_entropy, C

def calculate_gini(x):
    mp = Counter(x)
    total = len(x)
    p = 0
    for cnt in mp.values():
        p += math.pow(cnt / total, 2)
    gini_index = 1 - p
    return gini_index
        
def calculate_info_gain(x, conditions):
    entropy = calculate_entropy(x)
    condition_entropy = calculate_condition_entropy(x, conditions)
    info_gain = entropy - condition_entropy
    return info_gain

def split_with_infogain(x, conditions):
    feature_values = set(conditions[i] for i in range(len(conditions)))
    
    min_condition_entropy = float("inf")
    best_split = None
    
    for feature_value in feature_values:
        sub_x1 = x[conditions <= feature_value]
        entropy1 = calculate_entropy(sub_x1)
        condition_entropy1 = (len(sub_x1) / len(x)) * entropy1
        
        sub_x2 = x[conditions <= feature_value]
        entropy2 = calculate_entropy(sub_x2)
        condition_entropy2 = (len(sub_x2) / len(x)) * entropy2
        
        cur_condition_entropy = condition_entropy1 + condition_entropy2
        
        if cur_condition_entropy < min_condition_entropy:
            min_condition_entropy = cur_condition_entropy
            best_split = feature_value
    return best_split, min_condition_entropy

def split_with_gini(x, conditions):
    feature_values = set(conditions[i] for i in range(len(conditions)))
    
    min_gini = float("inf")
    best_split = None
    
    for feature_value in feature_values:
        sub_x1 = x[conditions <= feature_value]
        gini1 = calculate_gini(sub_x1) * (len(sub_x1) / len(x))
        
        sub_x2 = x[conditions <= feature_value]
        gini2 = calculate_gini(sub_x2) * (len(sub_x2) / len(x))
        
        cur_gini = gini1 + gini2
        
        if cur_gini < min_gini:
            min_gini = cur_gini
            best_split = feature_value
    return best_split, min_gini

class DiscTree():
    def __init__(self, node_type, category=None, feature=None):
        self.node_type = node_type
        self.category = category
        self.feature = feature
        
        self.dict = {}
        
    def add_node(self, edge, node):
        self.dict[edge] = node
        
    def predict(self, x):
        if self.node_type == "leaf":
            return self.category
        else:
            tree = self.dict[x[self.feature]]
            return tree.predict(x)
        
class ContTree():
    def __init__(self, node_type, category=None, feature=None, threshold=None):
        self.node_type = node_type
        self.category = category
        self.feature = feature
        self.threshold = threshold
        
        self.dict = {}
        
    def add_node(self, edge, node):
        self.dict[edge] = node
        
    def predict(self, x):
        if self.node_type == "leaf":
            return self.category
        else:
            if x[self.feature] <= self.threshold:
                treenode = self.dict["left"]
            else:
                treenode = self.dict["right"]
            
            return treenode.predict(x)
        
class DecisionTree():
    def __init__(self, epsilon=0.1, metric="id3", feature_type="discrete"):
        self.tree = None
        self.epsilon = epsilon
        self.metric = metric
        self.feature_type = feature_type
    
    def fit(self, X, y, features):
        if self.feature_type == "continuous":
            
            labels = set([y[i] for i in range(len(y))])
        
            if len(labels) == 1:
                tree = ContTree("leaf", labels.pop())
                self.tree = tree
                return tree
        
            max_class,max_len = max([(label,len(list(filter(lambda x:x==label, y)))) for label in labels],key = lambda x:x[1])
        
            if len(features) == 0: 
                tree = ContTree("leaf", max_class)
                self.tree = tree
                return tree
            
            if self.metric == "id3" or self.metric == "c4.5":
                max_info_gain = 0
                best_feature = None
                best_split = None
                
                entropy = calculate_entropy(y)
            
                for feature in features:
                    conditions = X[:, feature]
            
                    split_value, condition_entropy = split_with_infogain(y, conditions)
                
                    info_gain = entropy - condition_entropy
                
                    if info_gain > max_info_gain:
                        max_info_gain = info_gain
                        best_feature = feature
                        best_split = split_value
   
                if max_info_gain <= self.epsilon:
                    tree = ContTree("leaf", max_class)
                    self.tree = tree
                    return tree
                
            elif self.metric == "cart":
                max_gini_gain = 0
                best_feature = None
                best_split = None
                
                gini = calculate_gini(y)
                
                for feature in features:
                    conditions = X[:, feature]
            
                    split_value, condition_gini = split_with_gini(y, conditions)
                    
                    gini_gain = gini - condition_gini
                    
                    if gini_gain > max_gini_gain:
                        max_gini_gain = gini_gain
                        best_feature = feature
                        best_split = split_value
                    
                    if max_gini_gain <= self.epsilon:
                        tree = ContTree("leaf", max_class)
                        self.tree = tree
                        return tree
            
            else:
                raise NotImplementedError("The metric only includes id3, c4.5 and cart:)")
                    
            sub_features = list(filter(lambda x: x!=best_feature, features))
        
            tree = ContTree("non-leaf", feature=best_feature, threshold=best_split)
        
            feature_values = X[:, best_feature]
            feature_values = set([feature_values[i] for i in range(len(feature_values))])
            
            indices = []
            
            for i in range(len(y)):
                if X[i][best_feature] <= best_split:
                    indices.append(i)
                    
                    sub_X = X[indices]
                    sub_y = y[indices]
            
                    sub_tree = self.fit(sub_X, sub_y, sub_features)
            
                    tree.add_node("left", sub_tree)
                    
            for i in range(len(y)):
                if X[i][best_feature] > best_split:
                    indices.append(i)
                    
                    sub_X = X[indices]
                    sub_y = y[indices]
            
                    sub_tree = self.fit(sub_X, sub_y, sub_features)
            
                    tree.add_node("right", sub_tree)
                    
            self.tree = tree
            
            return tree
            
        else:
            assert self.metric == "id3" or self.metric == "c4.5", "The discrete features only support id3 or c4.5:("
            
            labels = set([y[i] for i in range(len(y))])
        
            if len(labels) == 1:
                tree = DiscTree("leaf", labels.pop())
                self.tree = tree
                return tree
        
            max_class,max_len = max([(label,len(list(filter(lambda x:x==label, y)))) for label in labels],key = lambda x:x[1])
        
            if len(features) == 0: 
                tree = DiscTree("leaf", max_class)
                self.tree = tree
                return tree

            max_info_gain = 0
            best_feature = None
            
            entropy = calculate_entropy(y)
            
            for feature in features:
                conditions = X[:, feature]

                condition_entropy, C = calculate_condition_entropy(y, conditions)
            
                info_gain = entropy - condition_entropy
            
                if self.metric == "c4.5":
                    info_gain /= C
            
                if info_gain > max_info_gain:
                    max_info_gain = info_gain
                    best_feature = feature
                    
            if max_info_gain <= self.epsilon:
                 tree = DiscTree("leaf", max_class)
                 self.tree = tree
                 return tree
             
            sub_features = list(filter(lambda x: x!=best_feature, features))

            tree = DiscTree("non-leaf", feature=best_feature)
        
            feature_values = X[:, best_feature]
            feature_values = set([feature_values[i] for i in range(len(feature_values))])

            for feature_value in feature_values:
                indices = []
                for i in range(len(y)):
                    if X[i][best_feature] == feature_value:
                        indices.append(i)
                    
                sub_X = X[indices]
                sub_y = y[indices]
            
                sub_tree = self.fit(sub_X, sub_y, sub_features)
            
                tree.add_node(feature_value, sub_tree)
            
                self.tree = tree
            
            return tree
    
    def predict(self, X):
        preds = []
        for x in X:
            pred = self.tree.predict(x)
            preds.append(pred)
        
        preds = np.array(preds)
        return preds
    
    def score(self, X, y):
        preds = self.predict(X)
        score = accuracy_score(y,preds)
        return score
        
if __name__ == "__main__":
    raw_data = load_iris()
    
    X = raw_data.data
    y = raw_data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=23323)
    
    dt = DecisionTree(metric="id3", feature_type="continuous")
    
    t = dt.fit(X_train[:10],y_train[:10],[i for i in range(4)])
    #print(dt.predict(X_train[:10]))
    #print(y_train[:10])
    #print(dt.score(X_test[:10], y_test[:10]))
    
    print(calculate_gini(y_train[0:20]))
    print(y_train[0:20])