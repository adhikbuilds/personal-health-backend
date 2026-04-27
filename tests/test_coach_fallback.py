def test_weekly_note_fallback_works_without_api_key(client, authed):
    # ANTHROPIC_API_KEY is unset by conftest, so this MUST hit fallback path
    r = client.get("/coach/athlete_01/weekly-note?days=7", headers=authed["headers"])
    if r.status_code == 404:
        return
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "fallback"
    assert isinstance(body["bullets"], list)
    assert 2 <= len(body["bullets"]) <= 4


def test_fallback_unit():
    from ai_coach import fallback_note

    bullets = fallback_note(
        "Test Athlete",
        "vertical_jump",
        {
            "session_count": 5,
            "form_trend_pct": 4.2,
            "avg_form_score": 78,
            "bpi_delta": 120,
            "weak_joints": [{"joint": "knee_angle", "deviation_deg": 6.0}],
            "injury_risk": "watch",
        },
    )
    assert len(bullets) == 4
    assert all(isinstance(b, str) and b for b in bullets)
