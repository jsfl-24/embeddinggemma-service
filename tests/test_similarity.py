import math


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


def test_similarity_sanity(client):
    dog_one = client.post(
        "/v1/embeddings", json={"input": "A dog is running in a park."}
    ).json()["embedding"]
    dog_two = client.post(
        "/v1/embeddings", json={"input": "A dog runs through a park."}
    ).json()["embedding"]
    database = client.post(
        "/v1/embeddings", json={"input": "The database server is unavailable."}
    ).json()["embedding"]

    similar = _cosine(dog_one, dog_two)
    unrelated = _cosine(dog_one, database)
    assert similar > unrelated