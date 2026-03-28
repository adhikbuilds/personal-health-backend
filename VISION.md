# Personal Health — The Full Vision

*For the team building this. Read this before you write a single line of code.*

---

## Why this exists

Most people who want to get seriously fit never will. Not because they lack discipline — but because they lack access. A good sports coach costs ₹2,000–₹5,000 per session. A biomechanics lab that can tell you your knee is caving inward at 23 degrees on your left side costs even more. Physiotherapists, nutritionists, recovery specialists — the entire infrastructure that elite athletes take for granted is locked behind money, geography, and who you know.

Your phone has a camera. That camera captures 30 frames per second. In each of those frames, there is enough information to tell someone — with the same precision as a biomechanics lab — whether their squat form is going to blow out their knee in six months, or whether they're on track to jump 60 centimetres.

That's what this is. We are building the coaching infrastructure that doesn't exist for the 99% of athletes who will never afford the 1%'s tools. The phone is the lab. The AI is the coach. The community is the motivation. And eventually, the model gets better every time someone trains.

---

## What we are building — in plain English

Three things. In this order.

**1. A data collection machine disguised as a training app**

The app does something genuinely useful on day one: it watches you train, grades your form, tells you what to fix, and tracks your progress over weeks. Athletes use it because it's actually good. But every session also generates labelled biomechanical data — joint angles, symmetry scores, phase transitions, heart rate — that no one has at scale from real athletes training in real conditions on phones.

**2. A model that gets smarter as more people use it**

Right now the model running on our backend was trained on synthetic data — mathematically grounded, sport-science calibrated, but synthetic. The more real sessions we collect, the better we can retrain it. Session by session, athlete by athlete, we close the gap between "synthetic" and "real" until the model is genuinely world-class. Then we put that model back on the phone.

**3. A coaching layer that speaks to you**

Once the model is good, we add a generative layer on top — an AI coach that reads your last week of sessions and tells you, in plain language, exactly what to work on this week, why, and how. Not a generic fitness plan. A personalised coaching note based on your actual biomechanics. This is the product that people pay for.

---

## The tech stack — what's already built

Here's where things stand right now. The foundation is real and working.

### Android App (`personal-health-android`)
React Native, Expo SDK 54. Runs in Expo Go — no build required.

- **TrainScreen**: Opens the back camera, captures frames at 5fps via `expo-camera`, sends them to the backend as base64 JPEG over HTTP. Gets back form scores, joint angles, feedback, phase labels. Displays them in real-time.
- **RPPGScreen**: Opens the front camera. Sends frames to the backend over WebSocket. Backend extracts average RGB channel values from the face region using Pillow and runs a photoplethysmography algorithm to estimate BPM and HRV. No wearable. No hardware. Just the front camera.
- **Constants**: `BACKEND_HOST` set in `app.json` under `expo.extra.backendHost`. Change your Wi-Fi IP there and the whole app points to your machine.

### Backend (`personal-health-backend`)
FastAPI, Python, MediaPipe, TensorFlow.

- **`api_server.py`**: The main API. Handles athlete registration, session start/end, frame ingestion (async queue so it never blocks the phone), leaderboard, active sessions, WebSocket for real-time dashboard push.
- **`pose_analyzer.py`**: Wraps MediaPipe Pose. Takes a base64 JPEG, runs landmark detection, computes all joint angles (knee L/R, hip L/R, elbow L/R, ankle dorsiflexion, trunk lean, spine deviation, shoulder–hip separation), runs the form quality classifier, returns phase label + feedback.
- **`rppg_processor.py`**: Takes a stream of RGB values from the face, applies bandpass filtering and peak detection to estimate heart rate and HRV.
- **`generate_dataset.py`**: Synthetic training data generator. 2,000 labelled frames across 5 sports (vertical jump, snatch, sprint, javelin, cricket batting). Each row has 23 biomechanical features and a quality label (elite / good / average / poor). Based on sport-science literature distributions.
- **`model_trainer.py`**: Trains a 3-layer MLP (23 → 64 → 32 → 4 classes). Exports `.h5` and `.tflite`. The `.tflite` is designed to eventually run on-device.
- **`seed_athletes.py`**: Generates 30 athletes across 5 tiers (Block, District, State, National, Elite) with realistic BPI scores and session histories.
- **`seed_sessions.py`**: Generates historical session data per athlete showing progressive improvement over 90 days.

### Frontend (`personal-health-frontend`)
Node.js, Express, EJS. Runs on port 8083 and proxies API calls to the FastAPI backend.

- **Dashboard**: Live session monitoring. Form score, joint angles, symmetry index updating in real-time via WebSocket. Leaderboard, athlete list, recent sessions table.
- **Map**: Leaflet.js map centered on the user's GPS location. Fetches nearby sports fields from the API. Shows athlete markers from the leaderboard.
- **Overview**: Landing page with real-time athlete and session counts, feature explainer, system status.

---

## The three layers — in more depth

### Layer 1: Data

Every training session captures, per frame:

| Signal | How we get it | Why it matters |
|--------|--------------|----------------|
| Joint angles (14 per frame) | MediaPipe Pose via back camera | Core biomechanics — form quality |
| Limb symmetry index | Left/right joint angle comparison | Injury risk, dominant-side bias |
| Phase label | Rule-based + classifier | Setup / descent / takeoff / flight / landing |
| Form score (0–100) | Trained MLP classifier | Summary quality signal |
| Estimated jump height | CoM trajectory from keypoints | Vertical power output |
| Heart rate (BPM) | rPPG via front camera | Aerobic load, recovery state |
| HRV | rPPG peak interval variance | Readiness, stress, overtraining signal |
| Session duration | Start/end timestamp | Volume proxy |
| XP | Weighted score formula | Gamification, engagement signal |

As users train more, we accumulate a dataset that is genuinely unique: real athletes, real conditions, real phones, real imperfect form. That dataset doesn't exist anywhere at scale. The synthetic data in `generate_dataset.py` is a placeholder until we have the real thing.

The moment we have 500 real sessions from real athletes, we retrain the model and it gets noticeably better.

### Layer 2: Intelligence

What the model needs to do, in order of complexity:

1. **Form quality classification** — already built. Given a single frame of joint angles, output elite/good/average/poor. Accuracy on synthetic test data: ~88%.

2. **Rep counting** — detect phase transitions (setup → descent → bottom → takeoff = 1 rep for jump, setup → down → up = 1 rep for squat). This is a sequence classification problem, not a frame-level one. Next to build.

3. **Weak point identification** — across a full session, which joint angle is consistently deviating from the ideal range for that sport/quality tier? "Your right hip angle at bottom position is averaging 118° when it should be 95° for your tier." This is just statistics over session frames, but surfaced intelligently.

4. **Injury risk flagging** — if symmetry index drops below 0.80 for three consecutive sessions, or a specific joint angle deviation is worsening week over week, flag it before it becomes an injury. This is where the app shifts from "training tool" to "health tool."

5. **Progressive load modelling** — given an athlete's performance trend, what volume and intensity should next week's training look like to continue improving without overreaching?

6. **On-device inference** — the `.tflite` export is already in place. The long-term goal is that the model runs entirely on the phone with no internet required. Privacy-preserving, works in areas with no signal, instant feedback.

### Layer 3: Generative

This is where the product gets differentiated. Anyone can show you a form score. Almost no one can tell you, in plain English, what to do about it.

The generative layer takes structured session data (form scores, joint deviations, HR trend, symmetry, volume) and produces:

- **Weekly coaching note**: "This week your left knee cave got worse on descent (avg 108° vs 102° last week). Two things: (1) Add 3 sets of banded clamshells every day. (2) Slow down your descent on the next session — try a 3-second eccentric. Your heart rate peaked at 178 BPM in sessions 2 and 3, which is high Zone 4 for your estimated max — consider one recovery day before Saturday."

- **Dynamic training plan**: Weekly plan generated based on your last 14 days of data. Adapts volume if you're overreaching, adds skill work if your form score has plateaued, reduces intensity if HRV suggests poor recovery.

- **Competition readiness score**: Composite metric — form trend (40%), symmetry trend (20%), volume consistency (20%), recovery state from HRV (20%). Tells an athlete whether they're ready to perform or need another training week.

This layer does not require training a model from scratch. It uses an LLM (Claude API, GPT-4, or similar) with structured session summaries as context. The model we train informs the data the LLM receives. The LLM provides the language.

---

## The go-to-market — how we actually get users

The app needs athletes to use it. Athletes need to feel like it's worth their time. Here's the sequence that makes that happen.

### Step 1: The Huddle

Before any ads, before any big launch — run a huddle. This is a 2-hour in-person training session with 10–20 athletes, a coach, and the app.

The format:
- Athletes warm up and train normally
- Everyone has the app open on their phone
- We run live sessions, show form scores on the dashboard in real-time on a laptop
- Coach narrates: "See how Rahul's left knee angle is 115° — that's why he's not getting full height"
- End with a group leaderboard reveal

Why this works: Athletes see the app do something real in front of them. They tell their training partners. The coach becomes an advocate. You get 10–20 real sessions worth of data in one afternoon.

Do this at 3–5 sports academies, universities, or gym communities before spending anything on ads.

### Step 2: The content loop

Every session the app produces shareable output. A form score card — athlete ID, sport, score, top feedback point, date — that looks clean enough to post.

The sharing flow:
1. Session ends
2. App generates a score card image (athlete avatar, BPI delta since last session, key metric like "jump height: 47cm")
3. One-tap share to Instagram Stories / WhatsApp

Athletes share because it's a flex. Coaches share because it demonstrates their athletes improving. The content is also the marketing.

This is how you get organic reach before you spend on ads.

### Step 3: Ads — what to run and where

Once you have 20–30 real video clips of athletes using the app (from the huddles), run:

**Instagram/Facebook:**
- Target: 16–30, interested in fitness, gym, sports, cricket/athletics
- Format: 15-second Reels showing the form score overlaid on real training footage
- Hook: "Your phone can tell you exactly what's wrong with your squat" + before/after of a form score improving
- CTA: "Download free — Expo Go"

**YouTube pre-roll:**
- Target: fitness/training/coaching channels
- Format: 30-second demo of the live form score updating in real-time
- Hook: "This is what ₹5,000/session coaching looks like. Except it's free."

**Do not run ads until:**
- You have real footage (not screen recordings)
- The app works reliably for a new user in under 2 minutes
- The backend stays up for at least 6 hours under light load

### Step 4: The coach acquisition channel

Athletes follow coaches. If a coach recommends something, the athlete uses it.

Target sports coaches at district and state academies. The pitch to coaches:
- "Your athletes can train on their own and you can see their form data from your phone"
- Show the dashboard — active sessions, form scores, leaderboard
- The coach becomes a hub: they set up the app for their 20 athletes, all 20 athletes generate 20x the data and referrals

Offer coaches early access and a "coach dashboard" view (this is just the existing web dashboard with a specific athlete list filtered view — no new build required).

---

## The data flywheel — why this compounds

This is the business model underneath everything:

```
More athletes use the app
        ↓
More real training sessions collected
        ↓
Retrain model on real data (better than synthetic)
        ↓
More accurate form scoring and feedback
        ↓
Athletes get better results
        ↓
Athletes tell other athletes
        ↓
More athletes use the app
```

Every athlete who trains with the app makes the app better for every future athlete. This is the moat. Generic fitness apps don't have this. Wearable companies have some of it, but only for the people who can afford wearables.

We have it for anyone with a smartphone.

---

## For the interns — what you are actually doing

You are not building a CRUD app. You are building a data collection and intelligence pipeline that compounds in value over time.

Every screen you build is a data collection surface. Every API endpoint is a pipeline stage. Every UX decision affects whether an athlete completes a session (and we get their data) or drops off (and we don't).

This means the quality of your work has a direct, compounding effect on the quality of the model we eventually train. A session that crashes halfway through = lost data. A form score that's visibly wrong = athlete stops trusting the app = no more sessions. A slow API that times out = athlete trains without the app.

The most important things to get right, in order:

1. **The session flow** — from "open app" to "form score on screen" in under 60 seconds, reliably, on a mid-range Android with variable Wi-Fi. This is the hardest and most important thing.

2. **The feedback loop** — the form score and coaching text need to feel correct to an athlete who knows their sport. If we tell a good squatter their form is "poor", they stop using the app. Calibrate against real athletes.

3. **The data capture** — every session needs to land in the database with all its frames, scores, and metadata intact. No silent failures. No dropped frames at high load. This is the whole point.

4. **The dashboard** — coaches and internal team need to see what's happening. Not pretty charts for the sake of charts — actual visibility into sessions, scores, trends.

5. **The training pipeline** — when we have real data, we need to be able to retrain the model in a day and redeploy. The scripts in `generate_dataset.py` and `model_trainer.py` are the blueprint. Keep them clean.

---

## What "done" looks like — milestones

**Milestone 1 — Working demo (now → 2 weeks)**
- App installs in Expo Go in under 5 minutes for a new user
- Session starts, form score appears within 3 seconds of the first frame
- Heart rate shows on RPPGScreen
- Dashboard reflects active session in real-time
- 5 real athletes have used it for at least one session each

**Milestone 2 — First huddle**
- 15+ athletes at a single training session using the app simultaneously
- Backend handles concurrent sessions without crashing
- Post-huddle: dashboard shows leaderboard from the session
- We have ≥150 real sessions in the database

**Milestone 3 — First retrain**
- 500+ real sessions collected
- `generate_dataset.py` supplemented with real session exports
- Model retrained on real + synthetic mix
- Form scoring accuracy visibly improves on real footage vs old model

**Milestone 4 — First coaching note**
- Weekly session summary API endpoint built
- LLM API integration for coaching note generation
- Athlete sees their weekly coaching note in the app
- This is the first thing people would pay for

**Milestone 5 — First revenue conversation**
- Monthly subscription: ₹199/month for unlimited sessions + weekly coaching
- Coach plan: ₹999/month for up to 25 athletes on their dashboard
- By this point the model is trained on real data, the product is genuinely useful, and the flywheel is turning

---

## What to run today

If you just cloned this repo and want to get the full stack running:

```bash
# 1. Start the backend
cd personal-health-backend
pip install -r requirements.txt
python seed_athletes.py        # populate 30 athletes
python seed_sessions.py        # populate 90 days of history
python api_server.py           # starts on :8082

# 2. Start the frontend
cd personal-health-frontend
npm install
node server.js                 # starts on :8083, proxies to :8082
# Open http://localhost:8083

# 3. Start the Android app
cd personal-health-android
# Edit app.json → expo.extra.backendHost → your machine's Wi-Fi IP
npm install
npx expo start
# Scan QR in Expo Go on your phone
```

For training data generation and model training (run this on a server with GPU):

```bash
cd personal-health-backend
python generate_dataset.py     # generates dataset/training_data.csv
python model_trainer.py        # trains model, exports .h5 and .tflite
```

---

## The one thing to hold onto

There are thousands of fitness apps. What makes this different is that it gets better with every session, it works for anyone with any phone, and it closes a gap that should not exist — between the athlete who can afford coaching and the one who can't.

That's worth building well.

---

*Last updated: March 2026*
*Questions: talk to the team.*
