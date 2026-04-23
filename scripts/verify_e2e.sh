#!/bin/bash
# Full end-to-end verification — exercises every major endpoint family
# the mobile + web apps depend on, as a freshly registered user.

set -u
BASE=http://localhost:8082
EMAIL="full_$(date +%s)@example.com"
EMAIL2="other_$(date +%s)@example.com"
P=0; F=0; N=0
ok()    { printf "  ✓ %s\n" "$*"; P=$((P+1)); }
fail()  { printf "  ✗ %s\n" "$*"; F=$((F+1)); }
note()  { printf "  · %s\n" "$*"; N=$((N+1)); }
sec()   { echo; echo "─── $* ────────────────────────────────────────────"; }

j()     { python3 -c "import sys,json; print(json.loads(sys.stdin.read())$1)" 2>/dev/null; }
has()   { python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print($1 in d)" 2>/dev/null; }
isarr() { python3 -c "import sys,json; d=json.loads(sys.stdin.read()); v=d.get('$1'); print(isinstance(v,list))" 2>/dev/null; }
status_of() { curl -sS -o /dev/null -w "%{http_code}" "$@"; }

sec "0. Health + meta"
H=$(curl -sS $BASE/health); [ -n "$H" ] && ok "/health" || fail "/health"
[ -n "$(curl -sS $BASE/livez)" ] && ok "/livez" || fail "/livez"
[ -n "$(curl -sS $BASE/readyz)" ] && ok "/readyz" || fail "/readyz"
[ "$(status_of $BASE/metrics)" = "200" ] && ok "/metrics" || fail "/metrics"
[ "$(status_of $BASE/banner)" = "200" ] && ok "/banner" || fail "/banner"
[ "$(status_of $BASE/)" = "200" ] && ok "GET /" || fail "GET /"

sec "1. Auth — register/login/me/refresh/logout"
R=$(curl -sS -X POST -H "Content-Type: application/json" -d "{\"email\":\"$EMAIL\",\"name\":\"Full Verify\",\"password\":\"Sup3rsecret!\"}" $BASE/auth/register)
USER_ID=$(echo "$R" | j '["user"]["id"]')
ATHLETE_ID=$(echo "$R" | j '["user"]["athlete_id"]')
TOKEN=$(echo "$R" | j '["access_token"]')
REFRESH=$(echo "$R" | j '["refresh_token"]')
[ -n "$USER_ID" ] && ok "register → user_id=$USER_ID" || fail "register"
[ -n "$ATHLETE_ID" ] && [ "$ATHLETE_ID" != "None" ] && ok "auto-provisioned athlete_id=$ATHLETE_ID" || fail "no athlete_id"
H="Authorization: Bearer $TOKEN"
ME=$(curl -sS -H "$H" $BASE/auth/me)
[ "$(echo $ME | j '["email"]')" = "$EMAIL" ] && ok "/auth/me email matches" || fail "/auth/me wrong"
LOG=$(curl -sS -X POST -H "Content-Type: application/json" -d "{\"email\":\"$EMAIL\",\"password\":\"Sup3rsecret!\"}" $BASE/auth/login)
[ -n "$(echo $LOG | j '["access_token"]')" ] && ok "/auth/login returns access_token" || fail "/auth/login"
NEW=$(curl -sS -X POST -H "Content-Type: application/json" -d "{\"refresh_token\":\"$REFRESH\"}" $BASE/auth/refresh)
[ -n "$(echo $NEW | j '["access_token"]')" ] && ok "/auth/refresh works" || fail "/auth/refresh"
[ "$(status_of -X POST -H "Content-Type: application/json" -d "{\"refresh_token\":\"$REFRESH\"}" $BASE/auth/logout)" = "204" ] && ok "/auth/logout 204" || fail "/auth/logout"

# Re-login since we just logged out the original refresh
R=$(curl -sS -X POST -H "Content-Type: application/json" -d "{\"email\":\"$EMAIL\",\"password\":\"Sup3rsecret!\"}" $BASE/auth/login)
TOKEN=$(echo "$R" | j '["access_token"]')
H="Authorization: Bearer $TOKEN"

sec "2. Athletes — list/get/create"
LIST=$(curl -sS $BASE/athletes)
[ "$(echo $LIST | isarr 'items')" = "True" ] && ok "/athletes has items[]" || fail "/athletes missing items[]"
[ "$(echo $LIST | isarr 'athletes')" = "True" ] && ok "/athletes has legacy athletes[] alias" || fail "/athletes missing athletes[]"

MINE=$(curl -sS $BASE/athlete/$ATHLETE_ID)
[ "$(echo $MINE | j '["id"]')" = "$ATHLETE_ID" ] && ok "/athlete/{id} returns my record" || fail "/athlete/{id}"

# Insights/coaching may or may not be implemented
CODE=$(status_of $BASE/athlete/$ATHLETE_ID/insights)
[ "$CODE" = "200" -o "$CODE" = "501" ] && ok "/athlete/{id}/insights returns $CODE" || fail "/athlete/{id}/insights got $CODE"

sec "3. Sessions — start/frame/latest/end/get/list/scorecard/replay/rep-count"
SS=$(curl -sS -X POST -H "$H" -H "Content-Type: application/json" -d "{\"athlete_id\":\"$ATHLETE_ID\",\"sport\":\"vertical_jump\"}" $BASE/session/start)
SID=$(echo $SS | j '["session_id"]')
[ -n "$SID" ] && ok "/session/start returns session_id" || fail "/session/start"

# minimal frame (no image_b64) — verifies the route is reachable
FR=$(curl -sS -X POST -H "$H" -H "Content-Type: application/json" -d '{"knee_angle_l":95,"knee_angle_r":95,"hip_angle_l":110,"hip_angle_r":110,"trunk_lean":5,"limb_symmetry_idx":0.97,"form_score":78,"form_quality":"good","primary_feedback":"steady","phase":"descent"}' $BASE/session/$SID/frame)
[ -n "$(echo $FR | j '["frame_num"]')" ] && ok "/session/{id}/frame accepts frame" || fail "frame"

LR=$(curl -sS $BASE/session/$SID/latest-result)
[ -n "$(echo $LR | j '["session_id"]')" ] && ok "/session/{id}/latest-result" || fail "latest-result"

EE=$(curl -sS -X POST -H "$H" $BASE/session/$SID/end)
[ "$(echo $EE | j '["xp_earned"]')" != "" ] && ok "/session/{id}/end returns summary" || fail "end summary"

GET=$(curl -sS $BASE/session/$SID)
[ "$(echo $GET | j '["session_id"]')" = "$SID" ] && ok "/session/{id} GET" || fail "session GET"

LISTS=$(curl -sS "$BASE/sessions?athlete_id=$ATHLETE_ID&limit=10")
[ "$(echo $LISTS | isarr 'sessions')" = "True" ] && ok "/sessions has sessions[]" || fail "sessions list"

SC=$(curl -sS $BASE/session/$SID/scorecard)
[ -n "$(echo $SC | j '["session_id"]')" ] && ok "/session/{id}/scorecard" || note "scorecard empty (no real frames captured)"

RC=$(curl -sS $BASE/sessions/$SID/rep-count)
[ -n "$(echo $RC | j '["session_id"]')" ] && ok "/sessions/{id}/rep-count" || note "rep-count not present"

REPLAY=$(curl -sS "$BASE/sessions/$SID/replay?downsample=2")
[ -n "$(echo $REPLAY | j '["session_id"]')" ] && ok "/sessions/{id}/replay" || note "replay empty"

sec "4. Daily tracker — write/read/history (owner only)"
W=$(curl -sS -X POST -H "$H" -H "Content-Type: application/json" -d '{"steps":1234,"active_minutes":15,"distance_km":1.2,"calories_burned":80,"calorie_intake":1500,"water_glasses":3,"sleep_hours":7.0,"date":"2026-04-23"}' $BASE/athlete/$ATHLETE_ID/daily-tracker)
[ "$(echo $W | j '["ok"]')" = "True" ] && ok "owner write" || fail "owner write: $W"
T=$(curl -sS -H "$H" $BASE/athlete/$ATHLETE_ID/daily-tracker)
[ "$(echo $T | j '["athlete_id"]')" = "$ATHLETE_ID" ] && ok "owner read tracker" || fail "owner read"
HIST=$(curl -sS -H "$H" $BASE/athlete/$ATHLETE_ID/daily-tracker/history?days=30)
[ "$(echo $HIST | isarr 'items')" = "True" ] && ok "history has items[]" || fail "history items"
[ "$(status_of $BASE/athlete/$ATHLETE_ID/daily-tracker)" = "401" ] && ok "anon read blocked (401)" || fail "anon read status mismatch"

sec "5. Fitness test — submit + history"
F1=$(curl -sS -X POST -H "$H" -H "Content-Type: application/json" -d "{\"athlete_id\":\"$ATHLETE_ID\",\"score\":72,\"level\":5,\"bmi\":22.0,\"sit_reach_cm\":20,\"run_600_seconds\":180,\"age_group\":\"Adult\"}" $BASE/fitness-test)
[ "$(echo $F1 | j '["athlete_id"]')" = "$ATHLETE_ID" ] && ok "POST /fitness-test" || fail "fitness submit: $F1"
FH=$(curl -sS $BASE/fitness-test/history/$ATHLETE_ID)
[ "$(echo $FH | isarr 'history')" = "True" ] && ok "/fitness-test/history" || fail "fitness history"

sec "6. Plan — weekly/regenerate/complete/history"
PL=$(curl -sS $BASE/plan/$ATHLETE_ID/weekly)
[ -n "$(echo $PL | j '["athlete_id"]')" ] && ok "/plan/{id}/weekly" || fail "plan weekly: $PL"
PR=$(curl -sS -X POST -H "$H" $BASE/plan/$ATHLETE_ID/regenerate)
[ -n "$PR" ] && ok "/plan/{id}/regenerate" || fail "plan regenerate"
PH=$(curl -sS $BASE/plan/$ATHLETE_ID/history?limit=3)
[ -n "$PH" ] && ok "/plan/{id}/history" || fail "plan history"

sec "7. Coach — broadcast/inbox/athletes (owner only)"
ROST=$(curl -sS $BASE/coach/$ATHLETE_ID/athletes)
[ "$(echo $ROST | isarr 'items')" = "True" ] && ok "/coach/{id}/athletes" || fail "coach roster"
# Need a second athlete to broadcast TO
R2=$(curl -sS -X POST -H "Content-Type: application/json" -d "{\"email\":\"$EMAIL2\",\"name\":\"Other\",\"password\":\"Sup3rsecret!\"}" $BASE/auth/register)
OTHER=$(echo $R2 | j '["user"]["athlete_id"]')
T2=$(echo $R2 | j '["access_token"]')
H2="Authorization: Bearer $T2"
BC=$(curl -sS -X POST -H "$H" -H "Content-Type: application/json" -d "{\"message\":\"e2e-marker\",\"athlete_ids\":[\"$OTHER\"]}" $BASE/coach/$ATHLETE_ID/broadcast)
[ -n "$(echo $BC | j '["id"]')" ] && ok "broadcast sent" || fail "broadcast: $BC"
INB=$(curl -sS $BASE/coach/inbox/athlete/$OTHER)
HAS=$(echo $INB | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); print(any('e2e-marker' in (b.get('message') or '') for b in d.get('broadcasts',[])))")
[ "$HAS" = "True" ] && ok "athlete inbox shows broadcast" || fail "inbox missing"
CINB=$(curl -sS $BASE/coach/$ATHLETE_ID/inbox)
[ "$(echo $CINB | isarr 'items')" = "True" ] && ok "coach inbox has items[]" || fail "coach inbox"

# Impersonation rejected
[ "$(status_of -X POST -H "$H2" -H "Content-Type: application/json" -d '{"message":"x"}' $BASE/coach/$ATHLETE_ID/broadcast)" = "403" ] && ok "broadcast impersonation 403" || fail "impersonation"

sec "8. Social — feed/leaderboard/creators/classes/playfields/follow/clap/map"
FEED=$(curl -sS $BASE/feed)
[ "$(echo $FEED | isarr 'items')" = "True" ] && ok "/feed has items[]" || fail "feed"
LB=$(curl -sS $BASE/leaderboard?limit=5)
[ "$(echo $LB | isarr 'items')" = "True" ] && ok "/leaderboard has items[]" || fail "leaderboard"
CR=$(curl -sS $BASE/creators/trending)
[ "$(echo $CR | isarr 'items')" = "True" ] && ok "/creators/trending has items[]" || fail "creators"
CL=$(curl -sS $BASE/classes)
[ "$(echo $CL | isarr 'items')" = "True" ] && ok "/classes has items[]" || fail "classes"
PF=$(curl -sS $BASE/playfields)
[ "$(echo $PF | j '["demo"]')" = "True" ] && ok "/playfields demo:true flag" || fail "playfields demo flag"

# Clap (owner-gated, idempotent)
CLP=$(curl -sS -X POST -H "$H" $BASE/athlete/$ATHLETE_ID/clap/$SID)
[ "$(echo $CLP | j '["count"]')" = "1" ] && ok "clap on own session count=1" || fail "clap"
[ "$(status_of -X POST -H "$H2" $BASE/athlete/$ATHLETE_ID/clap/$SID)" = "403" ] && ok "clap impersonation 403" || fail "clap impersonation"
[ "$(status_of $BASE/claps/$SID)" = "200" ] && ok "GET /claps/{tid} public" || fail "claps GET"

# Follow (write) needs auth — let me confirm
FOLL=$(curl -sS -X POST -H "$H" -H "Content-Type: application/json" -d "{\"follower\":\"$ATHLETE_ID\",\"following\":\"$OTHER\"}" $BASE/follow)
[ -n "$(echo $FOLL | j '["action"]')" ] && ok "/follow toggle" || fail "follow"

[ "$(status_of $BASE/map)" = "200" ] && ok "/map HTML" || fail "/map"

sec "9. Progress / Intelligence"
[ "$(status_of $BASE/athlete/$ATHLETE_ID/progress)" = "200" ] && ok "/athlete/{id}/progress" || fail "progress"
[ "$(status_of $BASE/weak-joints/$ATHLETE_ID)" = "200" ] && ok "/weak-joints/{id}" || fail "weak-joints"
[ "$(status_of $BASE/injury-risk/$ATHLETE_ID)" = "200" ] && ok "/injury-risk/{id}" || fail "injury-risk"
[ "$(status_of $BASE/readiness/$ATHLETE_ID)" = "200" ] && ok "/readiness/{id}" || fail "readiness"
[ "$(status_of "$BASE/athlete/$ATHLETE_ID/advanced-metrics?days=60")" = "200" ] && ok "/advanced-metrics" || fail "advanced-metrics"
[ "$(status_of $BASE/athlete/$ATHLETE_ID/weekly-summary)" = "200" ] && ok "/weekly-summary" || fail "weekly-summary"
[ "$(status_of $BASE/athlete/$ATHLETE_ID/load-recommendation)" = "200" ] && ok "/load-recommendation" || fail "load-recommendation"

sec "10. Coach intelligence"
WN=$(curl -sS $BASE/coach/$ATHLETE_ID/weekly-note)
[ -n "$(echo $WN | j '["athlete_id"]')" ] && ok "/coach/{id}/weekly-note" || fail "weekly-note: $WN"

sec "11. Huddles"
HU=$(curl -sS -X POST -H "$H" -H "Content-Type: application/json" -d "{\"name\":\"E2E Huddle\",\"sport\":\"vertical_jump\",\"coach_id\":\"$ATHLETE_ID\"}" $BASE/huddle/create)
HID=$(echo $HU | j '["huddle_id"]')
[ -n "$HID" ] && ok "/huddle/create" || note "huddle create returned: $HU"
if [ -n "$HID" ]; then
  J=$(curl -sS -X POST -H "$H" -H "Content-Type: application/json" -d "{\"athlete_id\":\"$ATHLETE_ID\"}" $BASE/huddle/$HID/join)
  [ -n "$J" ] && ok "/huddle/{id}/join" || fail "huddle join"
  [ "$(status_of $BASE/huddle/$HID/live)" = "200" ] && ok "/huddle/{id}/live" || fail "huddle live"
fi
[ "$(status_of $BASE/huddles)" = "200" ] && ok "/huddles list" || fail "huddles list"

sec "12. Admin (require admin role)"
[ "$(status_of -H "$H" $BASE/admin/export/sessions)" = "403" ] && ok "non-admin → /admin/export 403" || fail "admin export non-admin"
[ "$(status_of $BASE/admin/export/sessions)" = "401" ] && ok "anon → /admin/export 401" || fail "admin export anon"
[ "$(status_of $BASE/admin/export/stats)" = "200" ] && ok "/admin/export/stats public (no PII)" || fail "stats"

sec "13. Nutrition — fallback path"
NA=$(curl -sS -X POST -H "$H" -H "Content-Type: application/json" -d "{\"athlete_id\":\"$ATHLETE_ID\",\"image_b64\":\"AAAA\"}" $BASE/nutrition/analyze)
[ -n "$(echo $NA | j '["source"]')" ] && ok "/nutrition/analyze returned" || note "nutrition unreachable"

echo
echo "═════════════════════════════════════════════════"
echo "SUMMARY: $P passed · $F failed · $N notes"
echo "═════════════════════════════════════════════════"
exit $F
