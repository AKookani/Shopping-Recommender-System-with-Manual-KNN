import numpy as np
from collections import Counter

def euclidean_distance(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2, axis=1))

class KNearestNeighbors:
    def __init__(self, k=3):
        self.K = k

    def fit(self, initial_data, initial_labels):
        self.initial_data = np.array(initial_data)
        self.initial_labels = np.array(initial_labels)

    def predict(self, new_data):
        self.new_data = np.array(new_data)
        self.output_length = len(self.new_data)
        self.results = np.empty(self.output_length, dtype=int)
        
        for i in range(self.output_length):
            distances = euclidean_distance(self.initial_data, self.new_data[i])
            sorted_indices = np.argsort(distances)
            k_nearest_labels = self.initial_labels[sorted_indices][:self.K]
            self.results[i] = Counter(k_nearest_labels).most_common(1)[0][0]
        return self.results
