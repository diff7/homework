import numpy as np
from numpy.linalg import norm

class KNN:
    def __init__(self, normalize=True):
        self.normalize_data = normalize
        
    def normalize(self, data):
        if self.normalize_data : 
            return (data - data.mean())/data.std()
        else:
            return data
    
    def fit(self, data, targets):
        self.data_labeled = self.normalize(data)
        self.targets = targets
        
    def predict_proba(self, data_unlabaled, k):
        data_unlabaled = self.normalize(data_unlabaled)
        cosine_sims = np.dot(data_unlabaled, self.data_labeled.T) / \
            np.outer(norm(data_unlabaled, axis=1), norm(self.data_labeled, axis=1))
        #print(cosine_sims.shape)
        predictions = np.zeros(shape=(data_unlabaled.shape[0], len(set(self.targets))))
        
        for i in range(data_unlabaled.shape[0]):
            k_nearest = np.argsort(cosine_sims[i])[::-1][:k]
            for point_index in k_nearest:
                clas = self.targets[point_index]
                predictions[i,clas] += cosine_sims[i,point_index] # add distance because it is cosine from 0 - 1 
        
        return predictions
    
    
    def predict(self, data_unlabeled, k):
        return np.argmax(self.predict_proba(data_unlabeled, k), axis=1)