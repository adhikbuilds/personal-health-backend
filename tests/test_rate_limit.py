"""
Rate limiter tests.

The integration test goes via the in-process RateLimiter class so we
don't need to mutate global middleware state. The HTTP-level path is
covered by hammering a real endpoint with a tiny limiter directly.
"""

import time

from middleware import RateLimiter


def test_token_bucket_allows_burst_then_blocks():
    rl = RateLimiter(per_minute=60, burst=3)
    key = "1.2.3.4:test"
    # burst of 3 should pass
    for _ in range(3):
        ok, _, _ = rl.consume(key)
        assert ok
    # 4th in same instant should be blocked
    ok, remaining, retry_after = rl.consume(key)
    assert not ok
    assert remaining < 1
    assert retry_after > 0


def test_token_bucket_refills_over_time():
    rl = RateLimiter(per_minute=600, burst=2)  # 10/sec
    key = "5.6.7.8:test"
    rl.consume(key)
    rl.consume(key)
    # exhaust
    ok, _, _ = rl.consume(key)
    assert not ok
    time.sleep(0.25)  # ~2.5 tokens refilled
    ok, _, _ = rl.consume(key)
    assert ok


def test_independent_keys_independent_buckets():
    rl = RateLimiter(per_minute=60, burst=1)
    ok_a, _, _ = rl.consume("ip-a:auth")
    ok_b, _, _ = rl.consume("ip-b:auth")
    assert ok_a and ok_b
    blocked_a, _, _ = rl.consume("ip-a:auth")
    assert not blocked_a


def test_429_response_shape(client, monkeypatch):
    """End-to-end: shrink the global limiter and trip it."""
    import middleware as mw

    original = mw._LIMITER
    mw._LIMITER = RateLimiter(per_minute=60, burst=2)
    try:
        # /auth/me is not in the exempt list, so it counts
        for _ in range(3):
            client.get("/auth/me")  # 401s but still consume tokens
        r = client.get("/auth/me")
        assert r.status_code == 429
        body = r.json()
        assert body["error"] == "rate_limit_exceeded"
        assert "retry_after_seconds" in body
        assert "Retry-After" in r.headers
        assert r.headers.get("X-RateLimit-Remaining") == "0"
        assert r.headers.get("X-Request-ID")
    finally:
        mw._LIMITER = original
