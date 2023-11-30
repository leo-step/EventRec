import re

def parse(file_name):
    with open(file_name, "r") as fp:
        input_text = ''.join(fp.readlines())

    pattern = re.compile(r'>\s*(.*?)\s*<')

    matches = pattern.findall(input_text)

    matches = [match for match in matches if match != '']

    return matches

# print(parse("clubs.html"))