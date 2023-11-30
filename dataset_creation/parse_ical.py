import re
from datetime import datetime

def remove_parentheses(input_string):
    return re.sub(r'\([^)]*\)', '', input_string).strip()

def extract_events(file_name):
    with open(file_name, "r") as fp:
        text = ''.join(fp.readlines())
    pattern = re.compile(r'BEGIN:VEVENT\n(.*?)END:VEVENT', re.DOTALL)
    matches = re.findall(pattern, text)
    dt1 = datetime(2023, 8, 1, 8, 15, 17)
    dt2 = datetime(2023, 10, 31, 9, 30, 45)
    dt3 = datetime(2023, 11, 15, 9, 30, 45)
    dt4 = datetime(2024, 1, 22, 9, 30, 45)
    events_train = []
    events_val = []
    events_test = []
    # count_train = 0
    # count_val = 0
    # count_test = 0
    event_names = set()
    for match in matches:
        event_dict = {}
        lines = match.split('\n')
        for line in lines:
            split = line.split(':')
            key = split[0]
            if key == '':
                continue
            value = ':'.join(split[1:])
            event_dict[key] = value

        name = remove_parentheses(event_dict["SUMMARY;ENCODING=QUOTED-PRINTABLE"])
        if name in event_names:
            continue
        else:
            event_names.add(name)

        start = datetime.strptime(event_dict["DTSTART"], "%Y%m%dT%H%M%SZ")

        if dt1 <= start <= dt2:
            events_train.append(event_dict)
        elif dt2 < start <= dt3:
            events_val.append(event_dict)
        elif dt3 < start <= dt4:
            events_test.append(event_dict)

    return events_train, events_val, events_test

if __name__ == "__main__":
    import json

    events_train, events_val, events_test = extract_events("dataset_creation/ical_princeton.ics")

    events = {
        "train": events_train,
        "val": events_val,
        "test": events_test
    }

    with open("dataset/events.json", "w") as fp:
        json.dump(events, fp)