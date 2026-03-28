# Personal Health — Product Audit & Task Assignments

*PM-level audit. Written for the founding team and interns.*
*Based on full codebase review against VISION.md. March 2026.*

---

## How to read this document

Section 1 is the honest product scorecard — what we said we'd build vs what actually exists.
Section 2 is the RCA — why things are the way they are.
Section 3 is the task list, broken by intern with exact instructions.
Section 4 is model training — what the junior ML engineer needs to do step by step.
Section 5 is UI specs — what needs to change and exactly how.

---

# Section 1 — Product Scorecard

## What VISION.md promised vs what exists

| Feature | Promise | Reality | Gap |
|---------|---------|---------|-----|
| Real-time form scoring from phone camera | ✅ Promised | ✅ Works — MediaPipe + MLP scoring | Minor: trained on synthetic data only |
| Heart rate from front camera (no wearable) | ✅ Promised | ✅ Works — CHROM rPPG algorithm | Minor: warmup takes ~4s, signal noisy under bad lighting |
| Session tracking (frame-level data stored) | ✅ Promised | ⚠️ Partial — frames stored in memory, JSON on shutdown | DB resets wipe everything |
| Leaderboard with BPI | ✅ Promised | ✅ Works — athletes sorted by BPI, seeded | BPI only updates when XP added during session, not retroactively |
| Form score trend over time | ✅ Promised | ❌ Not built | No `/progress/:athlete_id` endpoint exists |
| Rep counting | ✅ Promised | ❌ Not built | Phase detection is rule-based if/else — no sequence model |
| Injury risk flagging | ✅ Promised | ❌ Not built | Symmetry data captured but never analysed across sessions |
| Weekly AI coaching note | ✅ Promised | ❌ Not built | No LLM integration, no summary endpoint |
| Progressive training plan | ✅ Promised | ❌ Not built | No load modelling logic anywhere |
| On-device inference (tflite) | ✅ Promised | ⚠️ Partial — .tflite exported but never loaded on phone | Backend does all inference |
| Video library per sport | ✅ Promised | ❌ UI stub only | 18 sports in AcademyScreen, no videos |
| Field booking | Mentioned | ❌ UI stub | Toast only, no backend endpoint |
| User authentication | Assumed | ❌ Not built | Everything runs as "athlete_01" |
| Historical trends / charts | ✅ Promised | ❌ MetricsScreen is 100% hardcoded | Spider chart has fake data |
| Ghost skeleton calibration | ✅ Promised | ✅ Works — GhostSkeletonScreen + /calibrate endpoint | Countdown UX is rough |
| Dashboard real-time monitoring | ✅ Promised | ✅ Works — WebSocket broadcast, active sessions | |
| Social feed | Nice-to-have | ⚠️ Stubbed — hardcoded posts, no real content | Entire social layer is mock |
| Map with nearby fields | ✅ Promised | ⚠️ Partial — map renders, playfields endpoint exists | `/playfields` endpoint not implemented in backend |
| Daily tracker (steps, water, calories) | Nice-to-have | ❌ Hardcoded values never update | Never connected to any data source |
| Classes / PE system | In-scope | ⚠️ Stubbed — data hardcoded, ratings not persisted | |

### Score: 4 fully working, 6 partial, 10 not built

---

## Feature benchmarking by user story

### User Story 1: "I want to train alone and know if my squat form is correct"

**Journey:** Open app → Train → See form score → Know what to fix

| Step | Status | Issue |
|------|--------|-------|
| Open TrainScreen | ✅ | |
| Select sport (Squat) | ❌ | Squat is NOT in the sport list. Only vertical_jump, snatch, sprint, javelin, cricket_bat. The dashboard dropdown has squat/push_up/pull_up but TrainScreen does not. |
| Start session | ✅ | |
| See live form score | ⚠️ | Shows simulated score for first 3–7 seconds, then real AI result. User can't tell the difference. |
| Understand feedback | ⚠️ | "Reduce trunk lean" is generic — same for every "average" form athlete. Not personalised. |
| Know improvement vs last session | ❌ | Not built. No history comparison. |

**RCA:** The core training loop works but the feedback layer is purely categorical ("average" = one of 3 generic strings). This is the most important feature and it's only halfway done.

---

### User Story 2: "I want to see my heart rate during training without a smartwatch"

**Journey:** Open RPPGScreen → Hold phone facing me → See BPM

| Step | Status | Issue |
|------|--------|-------|
| Open RPPGScreen | ✅ | |
| Camera loads | ✅ | |
| See BPM within 10 seconds | ⚠️ | Warmup takes ~4 seconds (20 frames at 5fps). Fine on a good day, but in low light or if face is slightly off-centre, quality stays at "fair" indefinitely. |
| Trust the reading | ⚠️ | No face detection indicator on the phone side. User doesn't know if their face is in frame. |
| See HRV and recovery readiness | ✅ | Clinical inference card shows readiness/fatigue/stress |
| Save the reading to their session | ❌ | HR data is lost when screen closes. Not stored to session. |

**RCA:** Algorithm is good. The problem is UX — no face detection feedback, and data not persisted.

---

### User Story 3: "I want to see my progress over the last month"

**Journey:** Open MetricsScreen → See form score trend, BPI growth, weak joints

| Step | Status | Issue |
|------|--------|-------|
| Open MetricsScreen | ✅ | |
| See real trends | ❌ | Every metric is hardcoded. Spider chart is fake. "Last 3 sessions" sparklines are fake. |
| See which joint is consistently weak | ❌ | Not built anywhere. |
| Get a recommendation | ❌ | Not built. |

**RCA:** MetricsScreen is pure design. No data pipeline behind it.

---

### User Story 4: "I'm a coach — I want to see how my athletes are doing without being there"

**Journey:** Open web dashboard → See active sessions → See form scores + joint angles

| Step | Status | Issue |
|------|--------|-------|
| Open dashboard | ✅ | |
| See who's training right now | ✅ | Active sessions panel auto-refreshes every 10s |
| See real-time form score | ✅ | WebSocket push works |
| Filter by athlete | ❌ | No athlete-specific view on dashboard |
| See trend over last week | ❌ | Not built |
| Export session data | ❌ | Not built |

**RCA:** Dashboard is good for live monitoring. Zero retrospective value.

---

### User Story 5: "I want to count my reps and know my form per rep"

**Journey:** Train → App counts reps → See form score per rep → Know which rep was bad

| Step | Status | Issue |
|------|--------|-------|
| Rep counting | ❌ | Not built. Phase detection exists (setup/descent/takeoff/flight/landing) but phase transitions are never counted. |
| Per-rep form score | ❌ | Not built — scoring is per-frame, not per-rep |
| Worst rep identified | ❌ | Not built |

**RCA:** This is the most valuable feature for an athlete and it's zero percent built. Needs a sequence model (LSTM or sliding window) to detect rep boundaries.

---

# Section 2 — Root Cause Analysis

### RCA 1: Why is so much data hardcoded?

The codebase was built fast. The API contracts were designed correctly — `/leaderboard`, `/athletes`, `/sessions` all exist and return real shapes. But the Android app's `src/data/constants.js` file was never wired up. It contains 400+ lines of hardcoded content (18 sports, nutrition plans, PE classes, feed posts, banners) that was meant to be a temporary stub but became permanent.

**Fix:** Each item in `constants.js` needs a corresponding backend endpoint and a one-time migration from constant → API call.

---

### RCA 2: Why is the model trained on synthetic data?

The training pipeline (`generate_dataset.py` → `model_trainer.py`) is complete and well-built. The problem is there are no real sessions. The synthetic data is mathematically sound (based on sports science literature distributions) but doesn't capture:

- Camera angle variations (phone held at different heights)
- Clothing obscuring joints
- Non-ideal lighting (gym overhead lights, outdoor)
- Athlete build variations (tall vs short changes landmark distances)
- Cultural movement patterns

**Fix:** Run 5 training huddles. Each huddle with 15 athletes × 3 sessions each = 45 sessions. At 100 frames per session = 4,500 real labelled frames. Enough to fine-tune significantly.

---

### RCA 3: Why is there no user auth?

The backend was built assuming a single athlete (`athlete_01`) for local development. Authentication was deprioritised. The consequence is all data is shared — every athlete ID is a string with no validation.

**Fix:** JWT tokens. Phone registers once, gets a token, sends it in every request header. Backend validates. 2-day build.

---

### RCA 4: Why does the map not show playfields?

The frontend `map.ejs` calls `GET /api/playfields?lat=X&lng=Y&radius=5` but this endpoint does not exist in `api_server.py`. The map renders (Leaflet + dark tiles) and geolocation works, but the field markers never load.

**Fix:** Either (a) add a `/playfields` endpoint that returns hardcoded fields near major cities, or (b) integrate the Google Places / Overpass API for real "sports ground" data. (a) is a 2-hour fix. (b) is 2 days.

---

### RCA 5: Why do MetricsScreen and HomeScreen show fake data?

Both screens were designed before the data pipeline was confirmed. The design is correct — the spider chart, sparklines, and daily tracker are the right UI for the data. But the data was never connected. `METRICS_DB` in `constants.js` has hardcoded values with fake `history: [72, 74, 75]` arrays.

**Fix:** Build a `/athlete/:id/metrics` endpoint that aggregates the last N sessions and computes averages per joint, form trend, BPI delta.

---

### RCA 6: Why is the dashboard/frontend UI rough?

The CSS was written correctly (dark theme, `#06b6d4` accent, Inter font). The main issue is the dashboard lacks any charts or visualisations — it's only text and numbers. The BPM, form score, and joint angles show as plain `<div>` elements. No spark charts, no history line, no visual.

The Android app's screens (HomeScreen especially) use many inline styles with inline colors and gradient hacks that don't compose cleanly. Several screens have inconsistent spacing.

---

### RCA 7: Why is there no squad/push_up/pull_up in TrainScreen?

The backend `StartSessionRequest` model accepts any `sport` string. The `pose_analyzer.py` SPORT_IDEAL_ANGLES dict has entries for `vertical_jump, snatch, sprint, javelin, cricket_bat`. Squat, push_up, pull_up are in the dashboard's sport dropdown but not in pose_analyzer. If you send `sport=squat` to the backend, the analyzer falls through to a generic scoring path.

**Fix:** Add squat, push_up, pull_up to `SPORT_IDEAL_ANGLES` in `pose_analyzer.py`, then add them to `TrainScreen` sport selection grid.

---

# Section 3 — Task Assignments

---

## TASK BLOCK A — Backend & Data
*Assigned to: Backend Intern*

### A1 — Add missing sports to pose analyzer
**File:** `personal-health-backend/pose_analyzer.py`
**What:** Add ideal angle ranges for `squat`, `push_up`, `pull_up` to `SPORT_IDEAL_ANGLES`
**Squat ideal angles:**
```python
"squat": {
    "knee_angle": (80, 110),       # deep squat: ~90° is ideal
    "hip_angle": (80, 105),
    "ankle_dorsiflexion": (75, 95),
    "trunk_lean": (0, 15),         # stay upright
    "symmetry": 0.90,
}
"push_up": {
    "elbow_angle": (85, 100),      # 90° at bottom
    "shoulder_angle": (35, 55),    # elbows 45° from body
    "trunk_lean": (0, 8),          # plank position — minimal lean
    "symmetry": 0.92,
}
"pull_up": {
    "elbow_angle": (10, 30),       # fully contracted at top
    "shoulder_angle": (160, 180),  # arms overhead
    "trunk_lean": (0, 10),
    "symmetry": 0.90,
}
```
**Done when:** `POST /session/start` with `sport=squat` returns valid form_score instead of generic scoring

---

### A2 — Add `/playfields` endpoint
**File:** `personal-health-backend/api_server.py`
**What:** Add endpoint that returns sports fields near a lat/lng
```python
@app.get("/playfields")
async def get_playfields(lat: float, lng: float, radius: float = 5.0):
    # For now: return hardcoded fields in major Indian cities
    # Phase 2: integrate Overpass API (free OpenStreetMap data)
    fields = [f for f in HARDCODED_FIELDS if haversine(lat, lng, f["lat"], f["lng"]) <= radius]
    return {"playfields": fields, "count": len(fields)}
```
Add 20 real playfields (stadiums, sports complexes, grounds) for major cities.
**Done when:** The Map page loads with at least 5 field markers visible near Bangalore/Mumbai/Delhi.

---

### A3 — Add `/athlete/:id/progress` endpoint
**File:** `personal-health-backend/api_server.py`
**What:** Aggregates session history for one athlete and returns a trend

```python
@app.get("/athlete/{athlete_id}/progress")
async def get_athlete_progress(athlete_id: str, days: int = 30):
    # Get all completed sessions for this athlete in the last N days
    # Return:
    # - form_score_trend: [{date, avg_form_score}] sorted by date
    # - bpi_trend: [{date, bpi}]
    # - weak_joints: {"knee_angle_l": avg_deviation, ...} sorted by deviation desc
    # - sessions_this_week: int
    # - improvement_vs_last_week: float (% change in avg form score)
    pass
```

**Done when:** MetricsScreen can call this endpoint and show real chart data.

---

### A4 — Add session timeout cleanup
**File:** `personal-health-backend/api_server.py`
**What:** Sessions that have been `active` for > 2 hours should be auto-ended
Add a background task that runs every 30 minutes and ends stale sessions.
**Done when:** Restarting the backend doesn't show 50 "active" sessions from yesterday.

---

### A5 — Fix session persistence (save on every write, not just shutdown)
**File:** `personal-health-backend/api_server.py`
**What:** Currently `_save_db()` is only called on shutdown. If the server crashes, all sessions since startup are lost.
Change: call `_save_db()` after every `session/end` and every `athlete` create/update.
**Done when:** Killing the server process (`Ctrl+C` or `kill`) and restarting shows all previously completed sessions in the dashboard.

---

### A6 — Wire `seed_athletes.py` and `seed_sessions.py` to startup
**File:** `personal-health-backend/api_server.py`
**What:** If `db/athletes.json` has fewer than 10 athletes on startup, auto-run the seed scripts.
This means interns don't have to remember to run seeds manually.
**Done when:** Deleting `db/athletes.json` and restarting brings up 30 athletes automatically.

---

## TASK BLOCK B — Android App
*Assigned to: Android Intern*

### B1 — Add Squat, Push Up, Pull Up to TrainScreen sport grid
**File:** `personal-health-android/src/screens/TrainScreen.js`
**What:** The sport selection grid currently shows 5 sports. Add 3 more.
Look for the `SPORTS` array or grid render and add:
```js
{ id: 'squat',   label: 'Squat',    icon: '🏋️', description: 'Lower body strength' },
{ id: 'push_up', label: 'Push Up',  icon: '💪', description: 'Upper body push' },
{ id: 'pull_up', label: 'Pull Up',  icon: '🔝', description: 'Upper body pull' },
```
**Done when:** User can select Squat/Push Up/Pull Up and get real form scoring.

---

### B2 — Add face detection indicator to RPPGScreen
**File:** `personal-health-android/src/screens/RPPGScreen.js`
**What:** Currently there's no feedback on whether the face is correctly in frame.
Add a coloured border ring around the camera view:
- Grey = no signal / waiting
- Yellow = face detected, signal collecting
- Green = signal good, reading reliable
Derive colour from the `signal_quality` field returned by the WebSocket.
**Done when:** User can see at a glance whether their face is positioned correctly.

---

### B3 — Persist heart rate data to session
**File:** `personal-health-android/src/screens/RPPGScreen.js`
**What:** When rPPG session ends (STOP button), POST the final BPM and HRV to:
```
POST /rppg/result
{ session_id, athlete_id, avg_bpm, hrv_ms, duration_seconds, quality }
```
Add this endpoint to `api_server.py` too (stores in athlete's session record).
**Done when:** After an rPPG scan, the HubScreen or MetricsScreen can show "Last measured HR: 72 BPM".

---

### B4 — Fix MetricsScreen: replace hardcoded data with real API call
**File:** `personal-health-android/src/screens/MetricsScreen.js`
**What:** Replace `METRICS_DB` with a call to `GET /athlete/:id/progress` (built in A3).
Map the response to the spider chart and sparkline format the screen already expects.
The screen already has the correct layout — it just needs real data.
**Done when:** After 3+ sessions, MetricsScreen shows actual joint angle averages and form trend.

---

### B5 — Fix daily tracker to use real step/calorie data
**File:** `personal-health-android/src/screens/HomeScreen.js`
**What:** Daily tracker (steps, water, calories, distance) is fully hardcoded.
- Steps: use `expo-sensors` `Pedometer` API — if not available, show "unavailable on this device"
- Water/calories: tap to manually log (store in AsyncStorage, reset at midnight)
- Remove the hardcoded progress bar values
**Done when:** Steps counter actually counts steps, water and calories are tap-to-increment.

---

### B6 — Add loading state when AI is processing (TrainScreen)
**File:** `personal-health-android/src/screens/TrainScreen.js`
**What:** When a frame is sent and the AI is processing (3-7 second gap), the UI should show:
- A small "AI analysing..." badge/pill in the bottom HUD
- The score field dims or shows a subtle pulse animation
- When real result arrives, the badge disappears and score updates
Currently there's no differentiation between simulated and real scores.
**Done when:** Users can tell when they're seeing a real AI score vs a simulated placeholder.

---

### B7 — Add rep counter to TrainScreen
**File:** `personal-health-android/src/screens/TrainScreen.js`
**What:** Use phase transitions returned by the backend to count reps.
The backend returns `phase` (setup / descent / takeoff / flight / landing) per frame.
Track phase transitions in the session:
- For squat/vertical_jump: `descent → takeoff` = 1 rep
- For push_up: `descent → ascent` = 1 rep
- For pull_up: `descent → ascent` = 1 rep
Display a rep counter in the bottom HUD (`REPS: 7`).
**Done when:** Doing 10 squats shows REPS: 10 at the end of the session.

---

## TASK BLOCK C — Frontend Dashboard
*Assigned to: Frontend Intern*

### C1 — Add chart library to dashboard
**File:** `personal-health-frontend/views/dashboard.ejs`
**What:** Add Chart.js (CDN) to the dashboard. This is needed for C2 and C3.
```html
<script src="https://cdn.jsdelivr.net/npm/chart.js@4/dist/chart.umd.min.js"></script>
```
Add a reusable `renderSparkline(canvasId, data, color)` helper in the dashboard script.
**Done when:** A small test line chart renders on the page.

---

### C2 — Add form score chart to live session panel
**File:** `personal-health-frontend/views/dashboard.ejs`
**What:** The "Live Session" panel currently shows 6 static metric values. Add a line chart below them:
- X axis: frame number (last 20 frames)
- Y axis: form score (0–100)
- Cyan line, no fill, minimal axes
- Updates in real-time as WebSocket frames arrive
Each time a new WebSocket message arrives, push the form_score to a rolling array (max 20 items) and call `chart.update()`.
**Done when:** Watching a live session shows a moving line chart of form quality over time.

---

### C3 — Add athlete progress view
**File:** `personal-health-frontend/views/dashboard.ejs`
**What:** Add a "Progress" button next to each athlete in the Athletes panel.
Clicking it opens a side panel (slide in from right) that shows:
- Athlete name and tier
- Form score trend (line chart, last 30 days, data from `/athlete/:id/progress`)
- Weak joint breakdown (horizontal bar chart — which joint is most off)
- Sessions this week vs last week
**Done when:** Clicking any athlete in the Athletes list shows their real progress data.

---

### C4 — Fix map `/playfields` endpoint call
**File:** `personal-health-frontend/views/map.ejs`
**What:** The map already calls `GET /api/playfields?lat=X&lng=Y&radius=5`. Once the backend endpoint (A2) is built, verify the response is correctly consumed.
Check that:
- `data.playfields[i].lat` and `.lng` exist (field marker loop uses `f.lat || f.latitude`)
- Marker popup shows field name and type
- Status shows "X fields nearby"
**Done when:** Map shows real sports field markers near the user's location.

---

### C5 — Clean up index.ejs mobile view
**File:** `personal-health-frontend/views/index.ejs` and `public/css/index.css`
**What:** On a phone browser (375px width), the navbar overlaps the hero text and the 3-column features grid is squashed.
Fixes:
- Navbar: hide nav-links on mobile, keep only brand + btn-nav
- Features grid: `grid-template-columns: 1fr` on mobile (already in CSS but verify)
- Stats row: wrap to 2 columns on mobile
- System grid: single column on mobile (already in CSS but verify)
**Done when:** Landing page looks clean on a 375px iPhone-sized browser window.

---

# Section 4 — Model Training Instructions

*For the ML intern. This is everything you need to know.*

---

## What the model does

The form quality classifier takes a 23-element vector of joint angles from a single frame and predicts one of four classes: `elite`, `good`, `average`, `poor`.

Architecture: `23 → 64 → 32 → 4` MLP (multi-layer perceptron). Exports as `.h5` (Keras) and `.tflite` (Android).

Current state: Trained on 2,000 synthetic frames. Test accuracy: ~88% on the synthetic test set. Real-world accuracy on actual athletes: estimated 65–75% (unknown — no benchmarking done yet).

---

## What you need to do — in order

### Step 1: Run existing pipeline, understand it

```bash
cd personal-health-backend

# Generate 2000 synthetic training samples
python generate_dataset.py
# Output: dataset/training_data.csv, dataset/sample_stats.json

# Train the model on synthetic data
pip install tensorflow scikit-learn pandas numpy
python model_trainer.py
# Output: models/pose_classifier.h5, models/pose_classifier.tflite, models/training_metrics.json

# Look at training_metrics.json — check accuracy per class
cat models/training_metrics.json
```

Check: What's the accuracy for `poor` vs `elite`? Are all four classes classified well, or is one class consistently confused with another?

---

### Step 2: Understand the feature vector

The model takes exactly 23 features. Open `feature_extractor.py` and `pose_analyzer.py` and find the 23 features. They are:

```
hip_angle_l, hip_angle_r,
knee_angle_l, knee_angle_r,
shoulder_angle_l, shoulder_angle_r,
elbow_angle_l, elbow_angle_r,
ankle_dorsiflexion_l, ankle_dorsiflexion_r,
trunk_lean, spine_deviation, shoulder_hip_sep, head_forward_pos,
com_height_norm, estimated_jump_height, limb_symmetry_idx,
[sport one-hot: 5 features → vertical_jump, snatch, sprint, javelin, cricket_bat]
[phase one-hot: 2 features → descent, takeoff (setup=baseline)]
```

Total: 17 biomechanical + 5 sport + 2 phase = 24 (verify in the code).

---

### Step 3: Expand the synthetic dataset

Run this improved data generation script — it generates 10,000 samples instead of 2,000, adds the 3 new sports, and adds noise profiles that better simulate real-world sensor variation:

```bash
python generate_dataset.py --samples 250 --output dataset/training_data_v2.csv
```

Edit `generate_dataset.py` to:
1. Change `SAMPLES_PER_SPORT_QUALITY = 100` to `250` (gives 5,000 rows)
2. Add `squat`, `push_up`, `pull_up` to `SPORT_DISTRIBUTIONS` (copy from the angle ranges in Task A1)
3. Add Gaussian noise injection: `±5% random noise` on all angle values to simulate measurement error
4. Add camera angle augmentation: randomly offset all angles by ±8° to simulate phone being held at different heights

---

### Step 4: Collect real data from athletes

This is the most important step. Every hour you spend on this is worth 10 hours of synthetic data improvement.

**What to collect:**

For each sport (start with `vertical_jump` and `squat` — highest demand):
- Film 5 different athletes
- Each athlete does 20 repetitions
- Capture at different phone positions (eye level, knee level, side angle)
- At least 2 athletes with "poor" form (beginner level is fine)
- At least 1 athlete with "elite" form (competitive level)

**How to label:**

A sports coach or you (after reading the angle ranges) labels each SESSION (not each frame) as elite/good/average/poor based on overall observation. The backend will auto-label each frame using the session label as a starting point.

**Export script — run this after each huddle:**

```bash
cd personal-health-backend
python export_real_data.py --output dataset/real_sessions_batch1.csv
```

(See the export script in Section 4 below — you need to write it.)

---

### Step 5: Mix real + synthetic, retrain

```bash
cd personal-health-backend

# Mix datasets (80% real, 20% synthetic for balance)
python mix_datasets.py \
  --real dataset/real_sessions_batch1.csv \
  --synthetic dataset/training_data_v2.csv \
  --output dataset/mixed_training.csv \
  --real-weight 4

# Retrain
python model_trainer.py --dataset dataset/mixed_training.csv --output models/v2/

# Evaluate
cat models/v2/training_metrics.json
```

Target: accuracy > 85% per class on a held-out real validation set.

---

### Step 6: Benchmark on held-out real data

Before deploying a new model version:
1. Hold out 20% of real session data as a test set (never train on it)
2. Run inference on every frame in the test set
3. Compare predicted quality vs labelled quality
4. Check the confusion matrix — is "elite" ever predicted as "poor"? That's the worst failure mode.

Acceptance criteria for deployment:
- Overall accuracy > 85%
- No "elite" frames predicted as "poor" (false negatives on form quality)
- "Poor" recall > 90% (we must catch bad form — better to flag too much than too little)

---

### Step 7: Deploy updated model

```bash
# Copy .tflite to Android assets
cp personal-health-backend/models/v2/pose_classifier.tflite \
   personal-health-android/assets/models/pose_classifier.tflite

# Restart backend (it loads .h5 on startup)
cd personal-health-backend
python api_server.py
```

---

## Data export script — write this

Create `personal-health-backend/export_real_data.py`:

```python
"""
Export completed sessions to a training CSV.
Labels each frame with the session-level quality label.

Usage:
  python export_real_data.py --quality good --output dataset/real_batch1.csv

The --quality flag applies a session-level label to all frames in the export.
Use this after a huddle where you know the overall form level of the athletes.
"""

import json, csv, argparse
from pathlib import Path

DB_PATH = Path(__file__).parent / "db"

def export(quality_label: str, output_path: str, min_frames: int = 30):
    with open(DB_PATH / "sessions.json") as f:
        sessions = json.load(f)

    rows = []
    for sid, session in sessions.items():
        if session.get("status") != "completed":
            continue
        frames = session.get("frames", [])
        if len(frames) < min_frames:
            continue

        for frame in frames:
            if not frame.get("pose_detected", False):
                continue

            row = {
                "session_id":             sid,
                "athlete_id":             session["athlete_id"],
                "frame_num":              frame.get("frame_num", 0),
                "sport":                  session["sport"],
                "hip_angle_l":            frame.get("hip_angle_l", 0),
                "hip_angle_r":            frame.get("hip_angle_r", 0),
                "knee_angle_l":           frame.get("knee_angle_l", 0),
                "knee_angle_r":           frame.get("knee_angle_r", 0),
                "shoulder_angle_l":       frame.get("shoulder_angle_l", 0),
                "shoulder_angle_r":       frame.get("shoulder_angle_r", 0),
                "elbow_angle_l":          frame.get("elbow_angle_l", 0),
                "elbow_angle_r":          frame.get("elbow_angle_r", 0),
                "ankle_dorsiflexion_l":   frame.get("ankle_dorsiflexion_l", 0),
                "ankle_dorsiflexion_r":   frame.get("ankle_dorsiflexion_r", 0),
                "trunk_lean":             frame.get("trunk_lean", 0),
                "spine_deviation":        frame.get("spine_deviation", 0),
                "shoulder_hip_sep":       frame.get("shoulder_hip_sep", 0),
                "head_forward_pos":       frame.get("head_forward_pos", 0),
                "com_height_norm":        frame.get("com_height_norm", 0.5),
                "estimated_jump_height":  frame.get("estimated_jump_height", 0),
                "limb_symmetry_idx":      frame.get("limb_symmetry_idx", 1.0),
                "form_score":             frame.get("form_score", 0),
                "phase_label":            frame.get("phase", "setup"),
                "quality_label":          quality_label,  # session-level label
                "feedback_tag":           frame.get("primary_feedback", ""),
                "source":                 "real",         # distinguish from synthetic
            }
            rows.append(row)

    if not rows:
        print(f"[EXPORT] No eligible frames found. Check that sessions are completed and have > {min_frames} frames.")
        return

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[EXPORT] {len(rows)} frames from {len(sessions)} sessions → {out}")
    print(f"[EXPORT] Quality label applied: {quality_label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quality", required=True, choices=["elite","good","average","poor"])
    parser.add_argument("--output", default="dataset/real_export.csv")
    parser.add_argument("--min-frames", type=int, default=30)
    args = parser.parse_args()
    export(args.quality, args.output, args.min_frames)
```

---

## Model training checklist (paste this into Notion/Linear)

```
[ ] Run generate_dataset.py with 250 samples per class — verify 6000+ rows in training_data_v2.csv
[ ] Add squat/push_up/pull_up to SPORT_DISTRIBUTIONS in generate_dataset.py
[ ] Run model_trainer.py on v2 dataset — check accuracy per class
[ ] Confuse matrix: elite vs good vs average vs poor — all must be > 80% precision
[ ] Run first huddle — collect 15+ athletes × 3 sessions each
[ ] After huddle: run export_real_data.py for each quality tier
[ ] Retrain on mixed dataset (real + synthetic)
[ ] Benchmark on held-out real test set
[ ] If accuracy > 85% per class: copy .tflite to Android assets and deploy
[ ] Document: which sports, how many athletes, accuracy per class, confusion matrix
```

---

# Section 5 — UI Improvement Specs

---

## Dashboard (Web)

**Current state:** Clean dark theme, correct data keys, metrics visible. Missing: any visualisation, charts, per-athlete drill-down.

**Changes needed:**

### 1. Add mini sparkline to each stat card
Each of the 4 stat cards (Active Sessions, Athletes, AI Model, Sessions Total) should have a 7-day sparkline below the number. Use Chart.js line chart, 20px height, single colour, no axes.

```
[ 3 ] Active Sessions
[___/\/\____]  ← 7-day trend, tiny
```

### 2. Form score chart in Live Session panel
Below the 6 metric boxes, add a `<canvas id="live-chart">` that shows the last 20 form scores as a line chart. Update on every WebSocket frame.

### 3. Replace text "Idle" badge with actual session state ring
Currently the session badge says "Idle" or "Live" in text.
Change to a coloured pulsing ring:
- Grey ring = idle
- Cyan pulsing ring = live session
- Green ring = session ended, results ready

### 4. Leaderboard row should link to athlete progress
Each row in the leaderboard should be clickable (cursor: pointer). On click, slide in the athlete progress panel (C3 task).

### 5. Add keyboard shortcut: `S` to start session, `E` to end
Small UX improvement for power users / coaches.

---

## Android App

**Current state:** 14 screens, polished visuals, good colour use. Problems: inconsistent spacing, HomeScreen feels cluttered, TrainScreen bottom HUD is dense.

**Changes needed:**

### 1. HomeScreen — reduce clutter
Current HomeScreen has: fitness score card + daily tracker + 8 content cards + AI banner + 4 feature banners + trending creators. That's 7 sections on one screen.

Remove or consolidate:
- "Featured banners" carousel (4 scrollable cards) → move to HubScreen
- "Trending creators" strip → move to SocialFeedScreen header
- "Content grid" (8 tiles) → move into AcademyScreen or GetActiveScreen

HomeScreen should have:
1. Greeting + fitness level band (today's state)
2. Daily tracker (4 metrics, tap to update)
3. Quick action bar: Train | Heart Rate | History (3 buttons)
4. Next session suggestion (based on last session's weak joint)

### 2. TrainScreen — separate "setup" from "live"
Currently setup (sport selection, pre-flight) and live session (camera, HUD) are on the same screen. The pre-flight checklist runs for a while then is dismissed.

Split into two states:
- `SETUP` state: full screen sport grid + pre-flight checklist
- `ACTIVE` state: full screen camera + minimal HUD
Add a visible transition between them (fade or slide).

### 3. TrainScreen HUD — reduce visual noise
Bottom HUD has: feedback text + form score + form quality badge + frame count + avg score + mode indicator + telemetry overlay. Too much.

Reduce to:
- Big form score number (primary — readable from arm's length)
- One line of feedback text below it
- Rep counter (once B7 is built)
- Stop button (always visible, top-right corner)

Move frame count, avg score, and mode indicator to a collapsible "debug" overlay (tap a small icon to expand).

### 4. RPPGScreen — add face guide overlay
Add a semi-transparent face oval guide (SVG or styled View) centered on the camera preview. The oval turns:
- Grey = waiting
- Yellow = face detected, calibrating
- Green = reading stable

This is the clearest UX signal for "am I positioned correctly."

### 5. Consistent card style across all screens
Currently cards use different border radii, shadow strengths, and padding values. Pick one set:
- Border radius: `16px`
- Border: `1px solid rgba(255,255,255,0.08)`
- Background: `rgba(255,255,255,0.04)`
- Padding: `16px`

Apply this to every card in HomeScreen, HubScreen, MetricsScreen, ClassesScreen.

### 6. Loading states
Every screen that makes an API call should show a skeleton loader (grey pulse rectangles) while loading, not a blank screen or spinner.

---

## Map (Web)

**Current state:** Dark Leaflet map, geolocation works, filter pills work. No actual field markers (endpoint missing).

**After A2 (playfields endpoint) is done:**
- Field markers should show a `🏟️` emoji in a styled div marker (already in map.ejs)
- Athlete markers should show the athlete's initials in a coloured circle
- Clicking a field marker should show a popup with: field name, sport type, "Get Directions" link to Google Maps

---

# Summary: What to build in what order

## Week 1 — Foundation fixes (all interns parallel)

| Task | Owner | Effort | Unblocks |
|------|-------|--------|---------|
| A1: Add squat/push_up/pull_up to pose_analyzer | Backend | 2h | B1 |
| A2: Add /playfields endpoint | Backend | 3h | C4 |
| A3: Add /athlete/:id/progress endpoint | Backend | 4h | B4, C3 |
| A5: Fix session persistence | Backend | 1h | Everything |
| B1: Add new sports to TrainScreen | Android | 2h | — |
| B6: Add AI processing state badge | Android | 2h | UX clarity |
| C1: Add Chart.js to dashboard | Frontend | 1h | C2, C3 |
| C2: Form score live chart | Frontend | 3h | Coach UX |

## Week 2 — Core features

| Task | Owner | Effort | Unblocks |
|------|-------|--------|---------|
| B4: MetricsScreen real data | Android | 4h | — |
| B7: Rep counter | Android | 6h | User story 5 |
| B3: Persist HR to session | Android + Backend | 3h | — |
| C3: Athlete progress view | Frontend | 4h | — |
| A4: Session timeout cleanup | Backend | 2h | — |
| ML Step 1–3: Expand dataset to 5000+ rows | ML | 4h | Model quality |

## Week 3 — Data collection (huddle prep + model training)

| Task | Owner | Effort |
|------|-------|--------|
| ML Step 4: Run first huddle, collect real sessions | Everyone | 1 day |
| ML Step 5: Mix datasets, retrain | ML | 4h |
| ML Step 6: Benchmark and validate | ML | 3h |
| B2: Face detection ring on RPPGScreen | Android | 3h |
| B5: Real step counter on HomeScreen | Android | 4h |

## Week 4 — Polish + prep for demo

| Task | Owner | Effort |
|------|-------|--------|
| Android UI consolidation (Section 5) | Android | 6h |
| C5: Mobile-responsive landing page | Frontend | 2h |
| A6: Auto-seed on startup | Backend | 1h |
| End-to-end test: open app → session → dashboard | Everyone | 2h |

---

*Total estimated to production-ready MVP: 3–4 weeks with 3 interns.*
*Prioritise: A1 → A3 → B4 → B7 → ML Step 4 (real data). Everything else is polish.*
