import json
from utils import *

def embed_people():
    with open("people.json", "r") as fp:
        people = json.load(fp)
    vectors = []
    for i, person_text in enumerate(people):
        if i % 10 == 1:
            print(f"{i}/{len(people)}")
        vectors.append(get_embedding(person_text))
    vectors = np.array(vectors)
    np.save("people_vectors.npy", vectors)

def embed_events():
    with open("events.json", "r") as fp:
        events = json.load(fp)
    vectors = []
    for i, event_dict in enumerate(events):
        if i % 10 == 1:
            print(f"{i}/{len(events)}")
        text = []
        bad_keys = set(["DTSTAMP", "LAST-MODIFIED", "CREATED", "SEQUENCE", "CONTACT", "DTSTART", "DTEND", "UID", "URL"])
        for key, value in event_dict.items():
            if key in bad_keys:
                continue
            text.append(value)
        text = '\n'.join(text)
        vectors.append(get_embedding(text))
    vectors = np.array(vectors)
    np.save("event_vectors.npy", vectors)

def create_dataset():
    # index of person: [event indexes]
    people_vecs = np.load("people_vectors.npy")
    event_vecs = np.load("event_vectors.npy")
    dataset = dict()

    for i, person_vec in enumerate(people_vecs):
        event_indexes = top_k_similar(person_vec, 10, event_vecs)
        dataset[i] = event_indexes.tolist()
    
    with open("dataset.json", "w") as fp:
        json.dump(dataset, fp)

if __name__ == "__main__":
    with open("people.json", "r") as fp:
        people = json.load(fp)
    with open("events.json", "r") as fp:
        events = json.load(fp)
    with open("dataset.json", "r") as fp:
        dataset = json.load(fp)

    print(people[20])
    print()

    recommended = get_results(events, dataset["20"])

    for event in recommended:
        print(event["SUMMARY;ENCODING=QUOTED-PRINTABLE"])
        print(event["DESCRIPTION"])
        print()