from majors import majors
from generate_people import generate_people
import json

people = []
for major in majors:
    print(major)
    descriptions = generate_people(major)
    while len(descriptions) != 5:
        print("regenerating")
        descriptions = generate_people(major)
    people.extend(descriptions)

with open("people.json", "w") as fp:
    json.dump(people, fp)