import json
from utils import *

with open("dataset/dataset.json") as fp:
    dataset = json.load(fp)

major_vectors = np.load("dataset/major_vectors.npy")
event_train_vectors = np.load("dataset/event_train_vectors.npy")
# event_val_vectors = np.load("dataset/event_val_vectors.npy")
# event_test_vectors = np.load("dataset/event_test_vectors.npy")


class MatrixModel:
    def __init__(self):
        self.dataset = dataset["train"]
        self.event_vectors = event_train_vectors

    def train(self):
        self.matrices = []
        for person in self.dataset:
            person_number = int(person)
            train_event_indexes = self.dataset[str(person_number)]
            train_matrix = self.event_vectors[train_event_indexes]
            self.matrices.append(train_matrix)

    def predict(self, test_vectors, k=5):
        predictions = []
        for matrix in self.matrices:
            predictions.append(np.max((test_vectors @ matrix.T), axis=1).argsort()[-k:][::-1])
        return predictions


class AlphaModel:
    def __init__(self, alpha=0.3):
        self.dataset = dataset["train"]
        self.event_vectors = event_train_vectors
        self.alpha = alpha

    def update(self, person_vec, event_vec, alpha):
        return (1-alpha)*person_vec + alpha*event_vec

    def train(self):
        self.vectors = major_vectors.copy()
        for person in self.dataset:
            person_number = int(person)
            train_event_indexes = self.dataset[str(person_number)]
            sorted_indexes = sorted(train_event_indexes)
            for event_idx in sorted_indexes:
                self.vectors[person_number] = self.update(self.vectors[person_number], self.event_vectors[event_idx], self.alpha)

    def predict(self, test_vectors, k=5):
        predictions = []
        for vector in self.vectors:
            predictions.append(top_k_similar(vector, k, test_vectors))
        return predictions


if __name__ == "__main__":
    val_vectors = np.load("dataset/event_val_vectors.npy")
    model = MatrixModel()
    model.train()
    print(model.predict(val_vectors, k=10))
