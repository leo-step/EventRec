import json
from utils import *
import os

def embed_people():
    with open("dataset/people.json", "r") as fp:
        people = json.load(fp)
    vectors = []
    for i, person_text in enumerate(people):
        if i % 10 == 1:
            print(f"{i}/{len(people)}")
        vectors.append(get_embedding(person_text))
    vectors = np.array(vectors)
    np.save("dataset/people_vectors.npy", vectors)

def embed_events():
    with open("dataset/events.json", "r") as fp:
        events = json.load(fp)
    for split in events:
        vectors = []
        for i, event_dict in enumerate(events[split]):
            if i % 10 == 1:
                print(f"{i}/{len(events[split])}")
            text = []
            bad_keys = set(["DTSTAMP", "LAST-MODIFIED", "CREATED", "SEQUENCE", "CONTACT", "DTSTART", "DTEND", "UID", "URL"])
            for key, value in event_dict.items():
                if key in bad_keys:
                    continue
                text.append(value)
            text = '\n'.join(text)
            vectors.append(get_embedding(text))
        vectors = np.array(vectors)
        np.save(f"dataset/event_{split}_vectors.npy", vectors)

def create_dataset():
    # index of person: [event indexes]
    people_vecs = np.load("dataset/people_vectors.npy")

    dataset = {
        "train": {},
        "val": {},
        "test": {}
    }

    for split in dataset:
        event_vecs = np.load(f"dataset/event_{split}_vectors.npy")
        k = 15 if split == "train" else 10
        for i, person_vec in enumerate(people_vecs):
            event_indexes = top_k_similar(person_vec, k, event_vecs)
            dataset[split][i] = event_indexes.tolist()
    
    with open("dataset/dataset.json", "w") as fp:
        json.dump(dataset, fp)

if __name__ == "__main__":
    create_dataset()
