import json
from utils import *
import pinecone

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
    def __init__(self, alpha=0.3, pinecone_index_name="alpha_model_index"):
        self.dataset = dataset["train"]
        self.event_vectors = event_train_vectors
        self.alpha = alpha
        self.pinecone_index_name = pinecone_index_name
        self.pinecone_index = None

    def update(self, person_vec, event_vec, alpha):
        return (1 - alpha) * person_vec + alpha * event_vec

    def create_pinecone_index(self):
        pinecone.init(api_key="")
        pinecone.create_index(index_name=self.pinecone_index_name, dimension=len(self.event_vectors[0]))

    def index_data(self):
        if not self.pinecone_index:
            self.create_pinecone_index()

        for person_num, train_event_indexes in self.dataset.items():
            vector_sum = self.vectors[person_num]
            for event_idx in train_event_indexes:
                vector_sum = self.update(vector_sum, self.event_vectors[event_idx], self.alpha)

            # Index the vector for the person
            self.pinecone_index.upsert(item_ids=[str(person_num)], vectors=[vector_sum])

    def train(self):
        self.vectors = major_vectors.copy()
        self.index_data()

    def predict(self, k=5):
        if not self.pinecone_index:
            self.create_pinecone_index()

        predictions = []

        for vector in self.vectors:
            query_result = self.pinecone_index.query(queries=[vector], top_k=k)
            similar_items = [result.id for result in query_result[0].matches]
            predictions.append(similar_items)

        return predictions

# class AlphaModel:
#     def __init__(self, alpha=0.3):
#         self.dataset = dataset["train"]
#         self.event_vectors = event_train_vectors
#         self.alpha = alpha

#     def update(self, person_vec, event_vec, alpha):
#         return (1-alpha)*person_vec + alpha*event_vec

#     def train(self):
#         self.vectors = major_vectors.copy()
#         for person in self.dataset:
#             person_number = int(person)
#             train_event_indexes = self.dataset[str(person_number)]
#             sorted_indexes = sorted(train_event_indexes)
#             for event_idx in sorted_indexes:
#                 self.vectors[person_number] = self.update(self.vectors[person_number], self.event_vectors[event_idx], self.alpha)

#     def predict(self, test_vectors, k=5):
#         predictions = []
#         for vector in self.vectors:
#             predictions.append(top_k_similar(vector, k, test_vectors))
#         return predictions


def recall(predictions, true):
    num_people = len(predictions)
    recall_sum = 0
    for i in range(num_people):
        pred = set(predictions[i])
        real = set(true[str(i)])
        num_correct = len(pred.intersection(real))
        total = len(real)
        recall_sum += num_correct / total
    return recall_sum / num_people


def precision(predictions, true):
    num_people = len(predictions)
    precision_sum = 0
    for i in range(num_people):
        pred = set(predictions[i])
        real = set(true[str(i)])
        num_correct = len(pred.intersection(real))
        k = len(predictions[0])
        precision_sum += num_correct / k
    return precision_sum / num_people


if __name__ == "__main__":
    val_vectors = np.load("dataset/event_val_vectors.npy")
    test_vectors = np.load("dataset/event_test_vectors.npy")

    # max_alpha = 0
    # max_rc = 0
    # for alpha in np.linspace(0, 0.5, num=50):
    #     model = AlphaModel(alpha=alpha)
    #     model.train()
    #     predictions = model.predict(val_vectors, k=10)
    #     rc = recall(predictions, dataset["val"])
    #     if rc > max_rc:
    #         max_alpha = alpha
    #         max_rc = rc

    # print(max_alpha, max_rc)

    model = MatrixModel()
    model.train()
    predictions = model.predict(val_vectors, k=10)
    print("MATRIX MODEL")
    print("VAL:", recall(predictions, dataset["val"]), precision(predictions, dataset["val"]))
    predictions = model.predict(test_vectors, k=10)
    print("TEST:", recall(predictions, dataset["test"]), precision(predictions, dataset["test"]))
    print()

    model = AlphaModel(alpha=0.1)
    model.train()
    predictions = model.predict(val_vectors, k=10)

    print("ALPHA MODEL (a=0.1)")
    print("VAL:", recall(predictions, dataset["val"]), precision(predictions, dataset["val"]))
    predictions = model.predict(test_vectors, k=10)
    print("TEST:", recall(predictions, dataset["test"]), precision(predictions, dataset["test"]))
    print()

'''

MATRIX MODEL
VAL: 0.38470588235294145 0.19235294117647073
TEST: 0.40000000000000024 0.20000000000000012

ALPHA MODEL (a=0.1)
VAL: 0.4741176470588236 0.2370588235294118
TEST: 0.5599999999999996 0.2799999999999998

'''