def test_request_id_echoed_back_when_provided(client):
    r = client.get("/", headers={"X-Request-ID": "test-fixed-id"})
    assert r.headers["X-Request-ID"] == "test-fixed-id"


def test_request_id_generated_when_missing(client):
    r = client.get("/")
    assert r.headers.get("X-Request-ID")


def test_unknown_route_404(client):
    r = client.get("/this/does/not/exist/at/all")
    assert r.status_code == 404
