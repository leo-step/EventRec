import json
from utils import *
from dataset_creation.majors import majors

with open("dataset/dataset.json") as fp:
    dataset = json.load(fp)
with open("dataset/events.json") as fp:
    events = json.load(fp)
with open("dataset/people.json") as fp:
    people = json.load(fp)
event_train_vectors = np.load("dataset/event_train_vectors.npy")
event_val_vectors = np.load("dataset/event_val_vectors.npy")

# all for person 0

person_number = 83

print(people[person_number])
print()

train_event_indexes = dataset["train"][str(person_number)]
train_events = get_results(events["train"], train_event_indexes)
# for event in train_events:
#     print(event["SUMMARY;ENCODING=QUOTED-PRINTABLE"])
#     print(event["DESCRIPTION"])
#     print()


train_matrix = event_train_vectors[train_event_indexes]
print(train_matrix.shape)
print(event_val_vectors.shape)
val_indexes = np.max((event_val_vectors @ train_matrix.T), axis=1).argsort()[-5:][::-1]
val_events = get_results(events["val"], val_indexes)
for event in val_events:
    print(event["SUMMARY;ENCODING=QUOTED-PRINTABLE"])
    print(event["DESCRIPTION"])
    print()
print("--------------")

val_event_indexes = dataset["val"][str(person_number)] # weird because a lot of office hours irrelevant crap
val_events = get_results(events["val"], val_event_indexes)
for event in val_events:
    print(event["SUMMARY;ENCODING=QUOTED-PRINTABLE"])
    print(event["DESCRIPTION"])
    print()

exit()

def update(person_vec, event_vec, alpha):
    return (1-alpha)*person_vec + alpha*event_vec

vector = np.load("initial_vectors.npy")[person_number]
# val_indexes = top_k_similar(vector, 10, event_val_vectors)

# val_events = get_results(events["val"], val_indexes)
# for event in val_events:
#     print(event["SUMMARY;ENCODING=QUOTED-PRINTABLE"])
#     print(event["DESCRIPTION"])
#     print()
# print("--------------")
# input()
# for each person, go thru their events in order, apply update rule to their person vector
events_indexes = sorted(train_event_indexes)
for event_idx in events_indexes:
    train_event = events["train"][event_idx]
    # print(train_event["SUMMARY;ENCODING=QUOTED-PRINTABLE"])
    # print(train_event["DESCRIPTION"])
    # print("^^^^^^")
    vector = update(vector, event_train_vectors[event_idx], 0.3)

# save trained people vectors
val_indexes = top_k_similar(vector, 5, event_val_vectors)

val_events = get_results(events["val"], val_indexes)
for event in val_events:
    print(event["SUMMARY;ENCODING=QUOTED-PRINTABLE"])
    print(event["DESCRIPTION"])
    print()
print("--------------")
# input()





# for event in events["val"]:
#     print(event["SUMMARY;ENCODING=QUOTED-PRINTABLE"])
#     print(event["DESCRIPTION"])
#     print()
# exit()

val_event_indexes = dataset["val"][str(person_number)] # weird because a lot of office hours irrelevant crap
val_events = get_results(events["val"], val_event_indexes)
for event in val_events:
    print(event["SUMMARY;ENCODING=QUOTED-PRINTABLE"])
    print(event["DESCRIPTION"])
    print()

# def update(person_vec, event_vec, alpha):
#     return (1-alpha)*person_vec + alpha*event_vec

# major_keywords = []
# for major in majors:
#     keyword = major.split(' ')[0]
#     major_keywords.append(keyword)
#     major_keywords.append(keyword.lower())
# # print(major_keywords)
# for i, person_text in enumerate(people):
#     # get initial vector (embed their major)
#     if i != 50:
#         continue
#     major = None
#     # text_set = set(person_text.split(' '))
#     for i, keyword in enumerate(major_keywords):
#         if keyword in person_text:
#             major = majors[i // 2]
#             break
#     if not major:
#         print(person_text)
#         raise ValueError("No major found")
#     print("MAJOR:", major)


