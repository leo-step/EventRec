import numpy as np
from openai import OpenAI

client = OpenAI(api_key="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")

# returns top k similar vector indexes
def top_k_similar(x, k, vecs):
    return (vecs @ x).argsort()[-k:][::-1]

# get OpenAI embedding
def get_embedding(text):
    response = client.embeddings.create(
        input=text,
        model="text-embedding-ada-002"
    )

    return np.array(response.data[0].embedding)

# select results given multiple indexes
def get_results(rows, indexes):
    return [rows[i] for i in indexes]