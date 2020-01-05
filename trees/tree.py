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
                predicted_class,
                predicted_proba):
   
        self.entropy = entropy
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_proba = predicted_proba
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
        predicted_proba = np.array(count_by_class)/m
        node = Node( ent, 
                    len(targets),
                    count_by_class,
                    predicted_class,
                    predicted_proba)
            
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
    
    def predict_proba(self, data):
        return np.array([self._predict_single_proba(row) for row in data])
    
    def _predict_single_proba(self, values):
        node = self.tree
        
        while node.left:
            if values[node.feature] > node.threshold:
                node = node.right
            else:
                node = node.left
        return node.predicted_proba
    
    
    
class RandomForest:
    def __init__(self, 
                 max_features = 0.7,
                 min_samples = 0.2,
                 max_depth = 6, # inclusive
                 num_trees=1,
                 min_depth = 4):   
        
        self.max_features = max_features
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.num_trees = num_trees
        self.min_depth = min_depth
    
    def plant_forest(self):
        self.forest =[]
        for i in range(self.num_trees):
            depth =  np.random.randint(self.min_depth,self.max_depth+1)
            tree = DecisionTree(maxdepth=depth)
            self.forest.append(tree)
        return self.forest
    
    def random_slice(self, data, targets):
        # random choice from  min sample size - max sample size 
        num_rows = np.random.randint(int(len(targets)*self.min_samples), targets.shape)
        row_index = np.random.choice(range(len(targets)), num_rows, replace=False)
        
        # get random features from min == 1 to max_features.
        num_features = np.random.randint(1, int(data.shape[1]*self.max_features))
        feature_index = np.random.choice(range(data.shape[1]), num_features, replace=False)
        return data[row_index][:,feature_index] , targets[row_index], feature_index
    
    def fit(self, data, targets):
        self.num_labels = len(set(targets))
        self.forest = self.plant_forest()
        self.trees_features = []
        for tree in self.forest:
            features_subset, targets_subset, features_index = self.random_slice(data, targets)
            self.trees_features.append(features_index)
            tree.fit(features_subset, targets_subset)
        
    def predict(self, features):
        preds = np.zeros(shape=(features.shape[0], self.num_labels))
        for tree, feature_mask in zip(self.forest, self.trees_features):
            preds+=tree.predict_proba(features[:,feature_mask])
        return np.argmax(preds / len(self.forest), axis=1)