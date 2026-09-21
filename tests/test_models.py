def test_model_info(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "embeddinggemma-300m"
    assert body["dimension"] == 768
    assert body["max_sequence_length"] == 2048
    assert body["runtime"] == "litert"