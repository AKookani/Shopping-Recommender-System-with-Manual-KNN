import numpy as np
from collections import Counter

def euclidean_distance(p, q):
    return np.sqrt(np.sum(np.array(p) - np.array(q)) ** 2)
    

class KNearestNeighbors:
    def __init__(self, k=3):
        self.K = k

    def fit(self, initial_data, initial_labels):
        self.initial_data = initial_data
        self.initial_labels = initial_labels
        self.data_length = len(self.initial_data)

    def predict(self, new_data):
        self.new_data = new_data
        self.output_length = len(self.new_data)
        self.distances = np.empty([self.data_length, 2], dtype=float)
        self.labels = np.empty(self.K, dtype=int)
        self.results = np.empty(self.output_length, dtype=int)
        
        for i in range(self.output_length):
            for j in range(self.data_length):
                distance = euclidean_distance(self.initial_data[j], self.new_data[i])
                self.distances[j, :] = np.array([distance, self.initial_labels[j]])
            sorted_indices  = np.argsort(self.distances[:, 0])
            for k, label in enumerate(self.distances[sorted_indices][:self.K]):
                self.labels[k] = label[1]
            self.results[i] = Counter(self.labels).most_common(1)[0][0]
        return self.results