# Personal Health Backend — Production Hardening Plan (v2, expanded)

*Author: Backend hardening pass, April 2026.*
*Scope: everything except model training (per instructions).*
*Skills applied (from `Orchestra-Research/AI-research-SKILLs` + Product-Manager-Skills):*
- `claude-api` — Anthropic SDK integration for the AI coach
- `problem-statement` — to frame each gap user-first
- `prd-development` — PRD-style acceptance criteria per endpoint
- `prioritization-advisor` — RICE-lite ordering of work
- `feature-investment-advisor` — kill/keep calls on speculative work
- `context-engineering-advisor` — how the AI coach prompt is built
- `langsmith` / `phoenix` (concept only) — tracing surface for future
- `instructor` / `outlines` (concept only) — structured AI outputs

---

## 1. Honest problem statement

> The Personal Health backend works in a **demo loop** but is **not safe to put in front of paying athletes**. Authentication does not exist — every API call is implicitly `athlete_01`. Logging is `print()`. Persistence is JSON files mutated in-process with no atomicity, no migrations, no audit trail. There is no rate limiting beyond a per-session frame guard, no global exception handler, no security headers, no observability surface (`/metrics`, request IDs, latency histograms), and no automated tests. Several VISION-promised endpoints (`/progress`, `/injury-risk`, weekly AI coaching note, weak-joint detection) **do not exist at all** — the AUDIT.md document calls them out as red. We cannot put this on the public internet, let alone charge ₹199/mo, until these gaps are closed in a single coherent pass.

### Who is blocked
- **Athletes** — cannot trust their own data is theirs, cannot see real progress, get only generic feedback.
- **Coaches** — cannot get athlete-scoped views, no auth means no privacy story.
- **Founders** — cannot publish a status page, debug an outage, or prove uptime.
- **Engineers** — no tests, no logs, no metrics → every change is a leap of faith.

---

## 2. Production scorecard (current → target)

| Capability | Current | Target | Risk if not done |
|---|---|---|---|
| Auth (register / login / JWT access + refresh) | ❌ none | ✅ bcrypt + JWT + refresh tokens in SQLite | CRITICAL — no privacy |
| Roles (athlete / coach / admin) | ❌ none | ✅ enum on user, dependency-based RBAC | HIGH |
| Password hashing | ❌ n/a | ✅ bcrypt cost 12 | CRITICAL |
| API key (service-to-service for dashboard) | ❌ none | ✅ table + middleware bypass | MEDIUM |
| Rate limiting (per-IP, per-route) | ❌ none | ✅ token bucket, configurable, 429 + Retry-After | HIGH |
| Structured logging + request IDs | ❌ `print()` | ✅ JSON logger + contextvar + access log middleware | HIGH |
| Global exception handler | ❌ tracebacks leak | ✅ sanitized JSON, error_id, logged with stack | HIGH |
| Security headers | ❌ none | ✅ HSTS, X-Frame, X-Content-Type, Referrer-Policy | MEDIUM |
| CORS env-driven (no wildcard in prod) | ⚠️ wildcard default | ✅ explicit list, credentials gated | MEDIUM |
| `/metrics` Prometheus exposition | ❌ none | ✅ counters + histograms + gauges, hand-rolled | MEDIUM |
| `/livez` + `/readyz` | ⚠️ only `/health` | ✅ split: liveness vs readiness with checks | LOW |
| Audit log (who did what when) | ❌ none | ✅ `audit_log` SQLite table, append-only | MEDIUM |
| Idempotency keys for write endpoints | ❌ none | ✅ `Idempotency-Key` header → cached response | MEDIUM |
| In-memory LRU cache (progress, leaderboard) | ❌ none | ✅ `cache.py` with TTL | LOW |
| Tests (pytest + httpx) | ❌ empty `tests/` | ✅ ≥15 tests across health, auth, progress, coach, rate-limit, middleware | HIGH |
| `/progress/{athlete_id}` (form trend, BPI growth) | ❌ missing (AUDIT red) | ✅ aggregates from real session frames | HIGH product gap |
| `/injury-risk/{athlete_id}` (symmetry stats) | ❌ missing (AUDIT red) | ✅ stats over `limb_symmetry_idx` | HIGH product gap |
| `/coach/weekly-note` (LLM coaching) | ❌ missing (AUDIT red) | ✅ Anthropic + deterministic fallback + retry + timeout | HIGH product gap |
| `/weak-joints/{athlete_id}` | ❌ missing | ✅ joint-deviation ranking | MEDIUM product gap |
| Daily tracker history endpoint | ⚠️ today only | ✅ 30-day series | LOW |
| Persistent SQLite for users/audit/cache | ❌ JSON only | ✅ `db/health.sqlite3` (additive — sessions stay JSON) | MEDIUM |
| Graceful shutdown ordering | ⚠️ partial | ✅ drain analysis queue, save db, close ws | LOW |
| OpenAPI tags + auth scheme advertised | ⚠️ partial | ✅ Bearer scheme in `/docs` | LOW |

---

## 3. Architecture decisions

### 3.1 Persistence: hybrid, additive
- Existing JSON files (`sessions.json`, `athletes.json`, `follows.json`) **stay**. Migrating them in this pass is too risky and the analysis worker depends on `SESSION_DB` being a live in-memory dict.
- New SQLite database `db/health.sqlite3` for **only the new concerns**: `users`, `refresh_tokens`, `api_keys`, `audit_log`, `idempotency_cache`, `progress_cache`, `daily_tracker_v2`. Hand-rolled tiny SQL — no SQLAlchemy dep, no Alembic complexity.
- Schema versioning via a `schema_version` row, migrations are top-down idempotent `CREATE TABLE IF NOT EXISTS` statements.

### 3.2 Auth model
- `POST /auth/register` → user row, returns access + refresh token.
- `POST /auth/login` → verify bcrypt, returns access (15 min) + refresh (30 days).
- `POST /auth/refresh` → rotate refresh token (single-use), returns new pair.
- `POST /auth/logout` → revoke refresh token row.
- `GET /auth/me` → current user from access token.
- Dependency `current_user` injects the user model into any route. `require_role("coach")` for RBAC.
- Backwards compatibility: existing endpoints stay unauthenticated for now (Android app needs to ship a token first). Auth dependencies are **opt-in per-route** to avoid breaking the demo. New endpoints (`/progress`, `/injury-risk`, `/coach/*`, `/auth/*`) accept an optional bearer; if absent, they fall back to query param `athlete_id` for dev.

### 3.3 Rate limiting
- In-process token-bucket keyed on `request.client.host` + route group.
- Default: 120 req/min, burst 30. Tunable per route via decorator.
- 429 response includes `Retry-After`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` headers.
- Skipped for `/livez`, `/readyz`, `/metrics`.

### 3.4 Observability
- Every request gets a UUID4 `request_id`, echoed in the `X-Request-ID` response header and every log line.
- JSON access log: method, path, status, latency_ms, request_id, ip, user_id (if known).
- `/metrics` returns Prometheus text format. Counters: `http_requests_total{route,method,status}`. Histogram: `http_request_duration_seconds_bucket`. Gauges: `analysis_queue_depth`, `active_sessions`, `ws_connections`. Hand-rolled — no `prometheus_client` dep.
- `/livez`: process is up. `/readyz`: DB readable + analysis queue initialized + < 90% queue full.

### 3.5 AI coach (using `claude-api` skill)
- Endpoint: `GET /coach/{athlete_id}/weekly-note`.
- Builds a structured prompt from the same data `/progress` returns (last 7 days form score trend, weak joints, injury risk, BPI delta). Prompt enforces a 4-bullet response: *what improved*, *what regressed*, *one drill*, *one warning*.
- Calls Anthropic via `anthropic` SDK on `claude-haiku-4-5-20251001`. Timeout 8s. 2 retries with exponential backoff. Circuit breaker: after 5 consecutive failures, breaker opens for 60s and we serve the deterministic template fallback.
- **Fallback (always available)**: if `ANTHROPIC_API_KEY` is unset OR breaker is open OR request fails, return a deterministic note built from the same stats. Endpoint is **never broken in dev**.
- Response always includes `source: "anthropic" | "fallback"` for transparency.
- Cost guard: cached for 1h per athlete via `progress_cache`.

### 3.6 New product endpoints (no model training)
- **`GET /progress/{athlete_id}?days=30`** — returns `form_score_trend[]`, `bpi_curve[]`, `weak_joints[]`, `session_count`, `total_reps`, `best_jump_cm`, `last_session_at`. All computed from existing `SESSION_DB` frames + scores. No new ML.
- **`GET /injury-risk/{athlete_id}?days=14`** — computes mean & stddev of `limb_symmetry_idx` across the window. Bands: `<5%` deviation = `low`, `5-12%` = `watch`, `>12%` = `high`. Returns reasoning + which side dominant.
- **`GET /weak-joints/{athlete_id}?days=30`** — for each tracked joint angle (hip, knee, shoulder, elbow, ankle, trunk), compute mean deviation from sport-ideal range. Returns top 3 weakest sorted by deviation.
- **`GET /athlete/{athlete_id}/daily-tracker/history?days=30`** — pulls the existing daily_tracker dict and returns it as a sorted time series with zero-fill.

---

## 4. File-by-file plan

### New files
```
personal-health-backend/
  PRODUCTION_PLAN.md           ← this file
  config.py                    ← typed settings
  logging_setup.py             ← JSON logger + request-id contextvar
  middleware.py                ← request-id, security headers, rate limit, exception handler, access log
  auth.py                      ← bcrypt + JWT + dependencies + RBAC
  cache.py                     ← in-process LRU with TTL
  metrics.py                   ← prom counters/histograms/gauges + /metrics text exposition
  sqlite_store.py              ← migrations + tiny CRUD helpers for users/audit/api_keys/etc
  ai_coach.py                  ← Anthropic client, retry, circuit breaker, deterministic fallback, prompt builder
  routes/auth.py               ← /auth/register, /login, /refresh, /logout, /me
  routes/progress.py           ← /progress, /injury-risk, /weak-joints
  routes/coach.py              ← /coach/{id}/weekly-note
  routes/admin.py              ← /livez, /readyz, /metrics, /audit (admin-only)
  tests/conftest.py
  tests/test_health.py
  tests/test_auth.py
  tests/test_progress.py
  tests/test_coach_fallback.py
  tests/test_rate_limit.py
  tests/test_middleware.py
```

### Edited files
- `api_server.py` — wire middleware, mount new routers, use JSON logger, configure CORS via settings, add OpenAPI bearer scheme.
- `requirements.txt` — add `bcrypt`, `pyjwt`, `anthropic`, `pytest`, `pytest-asyncio`.
- `.env.example` — add `JWT_SECRET`, `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL`, `RATE_LIMIT_PER_MINUTE`, `RATE_LIMIT_BURST`, `CORS_ORIGINS`, `ENV`, `JWT_TTL_MINUTES`.
- `routes/health.py` — keep existing `/health` but make it read from settings/sqlite.

### Files explicitly NOT touched
- `services/pose_analyzer.py`, `services/rppg_processor.py`, `services/realtime_analyzer.py`, `services/feature_extractor.py`, `services/intelligence.py`
- `pipeline/*` (model training)
- `routes/fitness.py` (live frame ingest path — too risky to refactor in this pass)
- `routes/social.py` (mock content stays mock)
- `models/*` (.tflite, .h5)

---

## 5. Acceptance criteria (each must be true)

1. `pytest -q` runs and **all new tests pass** with no network access (Anthropic mocked).
2. `curl -X POST /auth/register` creates a row in `db/health.sqlite3` and returns access + refresh tokens.
3. `curl -X POST /auth/login` followed by `curl /auth/me` with the bearer returns the user.
4. `curl /auth/me` **without** a bearer returns `401`.
5. Hammering any endpoint > burst returns `429` with `Retry-After`.
6. `curl /metrics` returns Prometheus text including `http_requests_total` after one prior request.
7. `curl /livez` is always 200; `curl /readyz` is 200 only after lifespan startup completes.
8. `curl /progress/athlete_01` returns real numbers from seeded sessions (form trend array length > 0).
9. `curl /injury-risk/athlete_01` returns one of `low|watch|high` with a reasoning string.
10. `curl /weak-joints/athlete_01` returns 3 joints sorted by deviation.
11. `curl /coach/athlete_01/weekly-note` returns a 4-bullet note **with `ANTHROPIC_API_KEY` unset** (fallback path) and source `"fallback"`.
12. Every response carries an `X-Request-ID` header.
13. Every error response is JSON `{error, error_id, request_id}` and the matching `error_id` appears in the JSON log.
14. CORS uses the env list, no wildcard if `ENV=prod`.
15. No `print()` calls in any new file; all use the JSON logger.

---

## 6. Prioritization (RICE-lite)

| Item | Reach | Impact | Confidence | Effort | Score | Order |
|---|---|---|---|---|---|---|
| Global exception handler + headers + request-id | every request | 5 | 5 | 1 | **125** | 1 |
| JSON logging | every request | 4 | 5 | 1 | **100** | 2 |
| Rate limit | every request | 4 | 4 | 1 | **64** | 3 |
| Auth + JWT + bcrypt + sqlite users | every user | 5 | 5 | 2 | **62.5** | 4 |
| `/progress` + `/injury-risk` + `/weak-joints` | every active athlete | 5 | 4 | 2 | **50** | 5 |
| `/metrics` + `/livez` + `/readyz` | ops | 3 | 5 | 1 | **75** | 6 |
| AI coach with fallback + circuit breaker | retained athletes | 5 | 3 | 2 | **37.5** | 7 |
| Tests | eng team | 4 | 5 | 2 | **50** | 8 |
| Audit log + idempotency + cache | future-proofing | 3 | 3 | 2 | **22.5** | 9 |

---

## 7. Out of scope (explicit, called out for next pass)

- Migrating `sessions.json` / `athletes.json` to SQLite — needs `SESSION_DB` rewrite + analysis worker refactor.
- Celery/Redis-backed background workers — current asyncio queue is fine until 10 RPS.
- Multi-tenant org/coach hierarchy — not needed for first revenue.
- WebSocket auth — Android app needs to ship a token first.
- Video upload & signed URLs — no video storage yet.
- Sentry / OTLP tracing exporter — `/metrics` is enough for now.
- Full e2e tests against a running uvicorn — TestClient is enough.

---

## 8. Commit & push

- Single conventional commit on the **`personal-health-backend`** repo only (`adhikbuilds/personal-health-backend`).
- Branch: whatever is currently checked out (do not switch).
- Subject: `feat(backend): production hardening — auth, middleware, observability, progress endpoints, AI coach`.
- No frontend / android push. No `personal-project` umbrella push.
- Push happens **only after** all acceptance criteria are met and tests are green.
