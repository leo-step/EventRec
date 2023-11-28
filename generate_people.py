# This code is for v1 of the openai package: pypi.org/project/openai
from openai import OpenAI
from majors import majors

client = OpenAI(api_key='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')

def generate_people(major):
    response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
        {
        "role": "system",
        "content": '''You are creating a database of Princeton students to create a event recommendation engine. This database will be used to match students to events.  Princeton students are very diverse. Many have passions outside of their academic major, and they vary in how involved they are with organizations. Make it representative of the Princeton student body.

    These are the possible interests you can choose from.

    ['Outdoor Activities', 'A Cappella', 'Academic', 'Advocacy &amp; Activism', 'Arts', 'Campus Engagement', 'Career Opportunity', 'Chaplaincies', 'Community Immerson', 'Criminal Justice', 'Cultural/Identity', 'Dance', 'Dialogues &amp; Partnerships', 'Educational', 'Entrepreneurship/Innovation', 'Environmental', 'Finance', 'Financial Literacy', 'Food', 'Games', 'Health', 'Hunger &amp; Homelessness', 'Immigration', 'Media', 'Music', 'Performing Arts', 'Political', 'Professional', 'Public Lectures', 'Publication', 'Racial Justice', 'Religious', 'Research', 'Service', 'Social', 'Social Justice', 'Special Interest', 'Sports/Recreation', 'Student Government', 'Sustained Volunteering']

    Each person is interested in 1-3 of these areas.'''
        },
        {
        "role": "user",
        "content": f'''Create a paragraph describing a Princeton student's interests and academic major of {major}. Some of their interests align with their major, other interests can be completely unrelated.

    Generate 5 students with a diverse range of interests, some of them can be less involved in the community than others. Format it into a numbered list, with a newline between the paragraphs. Make sure to have a unique name for each student.'''
        }
    ],
    temperature=1,
    max_tokens=2048,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
    )

    message_text = response.choices[0].message.content
    # print(message_text)
    people = message_text.split('\n\n')
    return people

# print(generate_people(majors[3]))