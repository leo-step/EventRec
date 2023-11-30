# take in dataset train in format of 0: [1, 2, 3, 4] that they went to
# make sure to sort events in chronological order (same as sorting indexes)
# feed events one by one into update formula
# initial vector is just major embedding

from utils import *
from majors import majors
import json
import os

def update(person_vec, event_vec, alpha):
    return (1-alpha)*person_vec + alpha*event_vec

def train(alpha=0.1):
    # initialize vectors for all the people first
    with open("dataset/people.json") as fp:
        people = json.load(fp)
    with open("dataset/dataset.json") as fp:
        dataset = json.load(fp)["train"]
    event_vectors = np.load("dataset/event_train_vectors.npy")

    if not os.path.exists("initial_vectors.npy"):
        people_vecs = []
        major_keywords = []
        for major in majors:
            keyword = major.split(' ')[0]
            major_keywords.append(keyword)
            major_keywords.append(keyword.lower())
        # print(major_keywords)
        for i, person_text in enumerate(people):
            # get initial vector (embed their major)
            if i % 10 == 1:
                print(i, "/", len(people))
            major = None
            # text_set = set(person_text.split(' '))
            for i, keyword in enumerate(major_keywords):
                if keyword in person_text:
                    major = majors[i // 2]
                    break
            if not major:
                print(person_text)
                raise ValueError("No major found")

            vector = get_embedding(major)
            people_vecs.append(vector)

        np.save("initial_vectors.npy", np.array(people_vecs))

    people_vecs = np.load("initial_vectors.npy")

    # for each person, go thru their events in order, apply update rule to their person vector
    for i in range(len(people_vecs)):
        events = dataset[str(i)][::-1]
        for event in events:
            people_vecs[i] = update(people_vecs[i], event_vectors[event], alpha)

    # save trained people vectors
    np.save("trained_vectors.npy", np.array(people_vecs))

def val():
    # read in trained_vectors
    # read in val dataset
    # read in val event vectors
    with open("dataset/people.json") as fp:
        people = json.load(fp)
    with open("dataset/events.json") as fp:
        events = json.load(fp)
    with open("dataset/dataset.json") as fp:
        dataset = json.load(fp)["val"]
    trained_vectors = np.load("trained_vectors.npy")
    event_vectors = np.load("dataset/event_val_vectors.npy")

    # for every person vector, find k=10 similar event vectors
    for vector in trained_vectors[20:]:
        print(people[20])
        event_indexes = top_k_similar(vector, 10, event_vectors)
        print(event_indexes)
        recommended = get_results(events["val"], event_indexes)
        for event in recommended:
            print(event["SUMMARY;ENCODING=QUOTED-PRINTABLE"])
            print(event["DESCRIPTION"])
            print()
        break
    # print(events)


if __name__ == "__main__":
    # val()
    # exit()
    initial_vectors = np.load("initial_vectors.npy")
    print(initial_vectors[20])
    trained_vectors = np.load("trained_vectors.npy")
    print(trained_vectors[20])
    truth = np.load("dataset/people_vectors.npy")
    print(truth[20])
    with open("dataset/events.json") as fp:
        events = json.load(fp)
    with open("dataset/people.json") as fp:
        people = json.load(fp)

    print(np.dot(initial_vectors[20], truth[20]), np.dot(trained_vectors[20], truth[20]))

    with open("dataset/dataset.json") as fp:
        dataset = json.load(fp)["train"]
        train_events = get_results(events["train"], dataset["20"])
        print(people[20])
        for event in train_events:
            print(event["SUMMARY;ENCODING=QUOTED-PRINTABLE"])
            print(event["DESCRIPTION"])
            print()