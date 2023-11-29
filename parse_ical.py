import re
from datetime import datetime

def extract_events(file_name):
    with open(file_name, "r") as fp:
        text = ''.join(fp.readlines())
    pattern = re.compile(r'BEGIN:VEVENT\n(.*?)END:VEVENT', re.DOTALL)
    matches = re.findall(pattern, text)
    dt1 = datetime(2023, 11, 28, 8, 15, 17)
    dt2 = datetime(2023, 12, 31, 9, 30, 45)
    events = []
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

        start = datetime.strptime(event_dict["DTSTART"], "%Y%m%dT%H%M%SZ")
        if dt1 <= start <= dt2:
            events.append(event_dict)
    return events

if __name__ == "__main__":
    import json

    events = extract_events("ical_princeton.ics")

    with open("events.json", "w") as fp:
        json.dump(events, fp)