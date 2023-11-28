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
    pass

if __name__ == "__main__":
    embed_people()