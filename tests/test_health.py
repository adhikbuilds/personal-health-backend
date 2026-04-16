def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "service" in r.json()
    assert "X-Request-ID" in r.headers


def test_livez(client):
    r = client.get("/livez")
    assert r.status_code == 200
    assert r.json()["status"] == "alive"


def test_readyz(client):
    r = client.get("/readyz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in {"ready", "not_ready"}


def test_metrics_text_format(client):
    client.get("/")  # generate one request first
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text
    assert "http_request_duration_seconds_bucket" in r.text


def test_security_headers(client):
    r = client.get("/")
    assert r.headers.get("X-Content-Type-Options") == "nosniff"
    assert r.headers.get("X-Frame-Options") == "DENY"
