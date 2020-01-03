from copy import copy
import numpy as np

class Stump:
    def __init__(self, num_classes):
        self.num_classes = num_classes
    
    def entropy(self, split, total):
            ent = 0
            for c in range(self.num_classes):
                if split[c] !=0:
                    p = split[c]/total
                    ent += -p*np.log(p)
            return ent
        
    def get_split(self, data, targets):
        #print(data)
        #print(targets)
        m = targets.size
        if  m <=1:
            return None, None
 
        data = np.array(data)
        targets = np.array(targets)
        
        count_by_class = [np.sum(targets==c) for c in range(self.num_classes)]
        best_ent = 0
        best_ent = -sum((n/m)*np.log(n/m) for n in count_by_class if n!=0)
        
        best_value, best_feature = None, None
        
        for feature in range(data.shape[1]):
            num_left = [0]*self.num_classes
            num_right = copy(count_by_class)
            values, classes = zip(*sorted((zip(data[:,feature], targets))))
            for idx in range(1,m):
                clas = classes[idx-1]
                num_left[clas]+=1
                num_right[clas]-=1
                #print(f'num_left {sum(num_left)}, num_right {sum(num_right)}')
                curr_ent = (idx*self.entropy(num_left, idx) + (m-idx)*self.entropy(num_right, m-idx))/m
                
                # if  two similar values are next to each other
                if values[idx]==values[idx-1]:
                    continue
                
                if curr_ent < best_ent:
                    best_ent = curr_ent
                    best_value  = (values[idx] + values[idx - 1]) / 2 
                    best_feature = feature
                
        return  best_value, best_feature
    
    
class Node:
    def __init__(self, 
                entropy, 
                num_samples, 
                num_samples_per_class, 
                predicted_class):
   
        self.entropy = entropy
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature = 0
        self.threshold = 0
        self.left = None
        self.right = None
        
        
class DecisionTree(Stump):
    def __init__(self, maxdepth=10):
        self.max_depth = maxdepth
        
    def fit(self, data, targets):
        num_classes = targets.max()+1
        Stump.__init__(self, num_classes)
        self.tree = self._grow_tree(data, targets, depth=0)
        
    
    def _grow_tree(self, data, targets, depth):
        m = len(data)
        count_by_class = [np.sum(targets==c) for c in range(self.num_classes)]
        ent = 0
        ent = -sum((n/m)*np.log(n/m) for n in count_by_class if n!=0)
        predicted_class = np.argmax(count_by_class)
        node = Node( ent, 
                    len(targets),
                    count_by_class,
                    predicted_class)
            
        if depth < self.max_depth:
            value, feature = self.get_split(data, targets)
            if feature is not None:
                node.threshold = value
                node.feature = feature
                indices_l = data[:,feature] < value
                d_left, t_left = data[indices_l], targets[indices_l]
                d_right, t_right = data[~indices_l], targets[~indices_l]
                node.left = self._grow_tree(d_left, t_left, depth+1)
                node.right = self._grow_tree(d_right, t_right, depth+1)
        return node
    
    
    def predict(self, data):
        return [self._predict_single(row) for row in data]
    
    def _predict_single(self, values):
        node = self.tree
        
        while node.left:
            if values[node.feature] > node.threshold:
                node = node.right
            else:
                node = node.left
        return node.predicted_class