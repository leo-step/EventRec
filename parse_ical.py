import re

def extract_events(file_name):
    with open(file_name, "r") as fp:
        text = ''.join(fp.readlines())
    pattern = re.compile(r'BEGIN:VEVENT\n(.*?)END:VEVENT', re.DOTALL)
    matches = re.findall(pattern, text)
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
        events.append(event_dict)
    return events

# print(extract_events("ical_princeton.ics")[0])