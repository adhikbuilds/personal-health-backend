"""
=============================================================================
ActiveBharat — Real-Time Pose Analysis (Webcam / USB Camera)
=============================================================================
Runs a live camera feed through MediaPipe BlazePose and overlays:
  - 33-keypoint skeleton
  - Real-time joint angles
  - Live form score and coaching cue
  - Phase indicator (descent/takeoff/flight/landing)
  - Session stats in corner HUD

Usage:
  python realtime_analyzer.py --sport vertical_jump [--camera 0]

Keyboard Controls:
  [q] Quit
  [r] Reset session / re-calibrate
  [s] Save current session to disk (JSON + CSV)
  [1-5] Switch sport (1=VJ, 2=Snatch, 3=Sprint, 4=Javelin, 5=Cricket)
=============================================================================
"""

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(__file__))

import argparse
import csv
import json
import uuid
from datetime import datetime
from pathlib import Path

try:
    import mediapipe as mp

    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("[WARNING] MediaPipe not installed. Run: pip install mediapipe")

from feature_extractor import FeatureExtractor
from pose_analyzer import BiomechanicalFrame, PoseAnalyzer

# ─── Constants ────────────────────────────────────────────────────────────────

SPORT_KEYS = {
    ord("1"): "vertical_jump",
    ord("2"): "snatch",
    ord("3"): "sprint",
    ord("4"): "javelin",
    ord("5"): "cricket_bat",
}

SPORT_DISPLAY = {
    "vertical_jump": "VERTICAL JUMP",
    "snatch": "OLYMPIC SNATCH",
    "sprint": "20m SPRINT",
    "javelin": "JAVELIN THROW",
    "cricket_bat": "CRICKET BAT",
}

QUALITY_COLORS = {
    "elite": (0, 255, 127),  # green
    "good": (0, 200, 255),  # cyan
    "average": (0, 165, 255),  # orange
    "poor": (0, 0, 255),  # red
    "unknown": (128, 128, 128),  # gray
}

# MediaPipe connector pairs for drawing skeleton
POSE_CONNECTIONS = [
    (11, 12),
    (11, 13),
    (13, 15),  # Left arm
    (12, 14),
    (14, 16),  # Right arm
    (11, 23),
    (12, 24),  # Torso sides
    (23, 24),  # Hips
    (23, 25),
    (25, 27),
    (27, 29),  # Left leg
    (24, 26),
    (26, 28),
    (28, 30),  # Right leg
    (0, 11),
    (0, 12),  # Head-shoulders
]

# ─── Overlay Drawing Helpers ─────────────────────────────────────────────────


def draw_skeleton(frame: cv2.Mat, lms, h: int, w: int, quality: str):
    color = QUALITY_COLORS.get(quality, (100, 100, 100))

    # Draw connections
    for a, b in POSE_CONNECTIONS:
        if lms[a].visibility > 0.5 and lms[b].visibility > 0.5:
            x1, y1 = int(lms[a].x * w), int(lms[a].y * h)
            x2, y2 = int(lms[b].x * w), int(lms[b].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    # Draw keypoints
    for i, lm in enumerate(lms):
        if lm.visibility > 0.5:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 4, color, -1, cv2.LINE_AA)
            cv2.circle(frame, (cx, cy), 6, (255, 255, 255), 1, cv2.LINE_AA)


def draw_angle_label(frame: cv2.Mat, lm, h: int, w: int, angle: float, label: str):
    """Draw an angle annotation near a joint."""
    if lm.visibility < 0.5 or angle < 1:
        return
    cx, cy = int(lm.x * w) + 10, int(lm.y * h) - 10
    text = f"{angle:.0f}°"
    cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)


def draw_hud(frame: cv2.Mat, bio: BiomechanicalFrame, sport: str, session_frames: int, avg_score: float):
    h, w = frame.shape[:2]
    quality = bio.form_quality
    q_color = QUALITY_COLORS.get(quality, (128, 128, 128))

    # Semi-transparent top bar
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 85), (15, 15, 35), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    # Sport label
    cv2.putText(
        frame, SPORT_DISPLAY.get(sport, sport), (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (6, 182, 212), 1, cv2.LINE_AA
    )

    # Form score
    score_text = f"FORM: {bio.form_score:.0f}%"
    cv2.putText(frame, score_text, (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, q_color, 2, cv2.LINE_AA)

    # Quality badge
    cv2.putText(frame, quality.upper(), (w - 100, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, q_color, 2, cv2.LINE_AA)

    # Phase indicator
    cv2.putText(
        frame, f"PHASE: {bio.phase.upper()}", (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA
    )

    # Symmetry
    sym_color = (0, 255, 127) if bio.limb_symmetry_idx > 0.9 else (0, 165, 255)
    cv2.putText(
        frame,
        f"SYM: {bio.limb_symmetry_idx:.2f}",
        (w - 120, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        sym_color,
        1,
        cv2.LINE_AA,
    )

    # Jump height
    if bio.estimated_jump_height > 5:
        cv2.putText(
            frame,
            f"EST VJ: {bio.estimated_jump_height:.1f}cm",
            (w - 150, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 200, 0),
            1,
            cv2.LINE_AA,
        )

    # Bottom coaching bar
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 60), (w, h), (15, 15, 35), -1)
    cv2.addWeighted(overlay2, 0.85, frame, 0.15, 0, frame)

    feedback = bio.primary_feedback or "Analyzing..."
    fb_color = (0, 100, 255) if bio.form_score < 75 else (0, 200, 100)
    cv2.putText(frame, feedback, (12, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, fb_color, 1, cv2.LINE_AA)
    cv2.putText(
        frame,
        f"FRAMES: {session_frames} | AVG: {avg_score:.1f}%",
        (12, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (150, 150, 150),
        1,
        cv2.LINE_AA,
    )

    # Visibility warning
    if not bio.visibility_ok:
        cv2.putText(
            frame, "⚠ SUBJECT OUT OF FRAME", (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA
        )

    # Controls hint
    cv2.putText(
        frame,
        "[q]Quit [r]Reset [s]Save [1-5]Sport",
        (10, h - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.28,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )


# ─── Session Manager ─────────────────────────────────────────────────────────


class SessionManager:
    def __init__(self, output_dir: str = "sessions"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.extractor = FeatureExtractor(normalize=False)
        self._records = []

    def add_frame(self, bio: BiomechanicalFrame, sport: str, session_id: str, frame_num: int):
        record = self.extractor.frame_to_record(bio, sport, session_id, frame_num)
        self._records.append(record)

    def save(self, analyzer: PoseAnalyzer, session_id: str, athlete_id: str = "athlete_01"):
        """Save current session as CSV + JSON summary."""
        if not self._records:
            print("[SAVE] No frames to save.")
            return

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save CSV
        csv_path = self.output_dir / f"session_{ts}_{session_id[:8]}.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.extractor.get_csv_headers())
            writer.writeheader()
            writer.writerows(self._records)
        print(f"[SAVE] CSV saved: {csv_path}")

        # Save summary JSON
        summary = analyzer.get_session_summary(athlete_id, session_id)
        json_path = self.output_dir / f"summary_{ts}_{session_id[:8]}.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[SAVE] Summary saved: {json_path}")

        self._records.clear()
        print(f"[SAVE] Peak form: {summary.get('peak_form_score', 0)}% | VJ: {summary.get('peak_jump_height_cm', 0)}cm")


# ─── Main Entry Point ─────────────────────────────────────────────────────────


def run_analyzer(sport: str = "vertical_jump", camera_id: int = 0, athlete_id: str = "athlete_01"):
    if not MP_AVAILABLE:
        print("[ERROR] Please install MediaPipe: pip install mediapipe opencv-python")
        return

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # 0=lite, 1=full, 2=heavy
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_id}")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    analyzer = PoseAnalyzer(sport=sport)
    session_manager = SessionManager()
    session_id = str(uuid.uuid4())
    frame_num = 0
    scores = []

    print(f"[START] Session: {session_id[:8]} | Sport: {sport}")
    print("[CONTROLS] [q]=Quit [r]=Reset [s]=Save [1-5]=Switch Sport")

    while True:
        ret, frame_img = cap.read()
        if not ret:
            break

        frame_img = cv2.flip(frame_img, 1)  # Mirror for natural feedback
        h, w = frame_img.shape[:2]

        # Run MediaPipe
        rgb = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = pose.process(rgb)
        rgb.flags.writeable = True

        # Analyze
        bio = analyzer.analyze(results)
        scores.append(bio.form_score)
        if len(scores) > 180:
            scores.pop(0)
        avg_score = sum(scores) / len(scores) if scores else 0

        # Draw skeleton overlay
        if results.pose_landmarks and bio.visibility_ok:
            draw_skeleton(frame_img, results.pose_landmarks.landmark, h, w, bio.form_quality)

            lms = results.pose_landmarks.landmark
            # Draw key angles
            draw_angle_label(frame_img, lms[25], h, w, bio.knee_angle_l, "L_KNEE")
            draw_angle_label(frame_img, lms[26], h, w, bio.knee_angle_r, "R_KNEE")
            draw_angle_label(frame_img, lms[23], h, w, bio.hip_angle_l, "L_HIP")
            draw_angle_label(frame_img, lms[24], h, w, bio.hip_angle_r, "R_HIP")

        # Draw HUD
        draw_hud(frame_img, bio, analyzer.sport, frame_num, avg_score)

        # Save frame data
        session_manager.add_frame(bio, analyzer.sport, session_id, frame_num)
        frame_num += 1

        cv2.imshow("ActiveBharat — Vision Engine", frame_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            print("[RESET] New session started")
            analyzer = PoseAnalyzer(sport=analyzer.sport)
            session_id = str(uuid.uuid4())
            frame_num = 0
            scores.clear()
        elif key == ord("s"):
            session_manager.save(analyzer, session_id, athlete_id)
        elif key in SPORT_KEYS:
            new_sport = SPORT_KEYS[key]
            analyzer.set_sport(new_sport)
            print(f"[SPORT] Switched to: {new_sport}")

    cap.release()
    cv2.destroyAllWindows()
    pose.close()

    # Auto-save on exit
    if frame_num > 0:
        session_manager.save(analyzer, session_id, athlete_id)
    print("[END] Session complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ActiveBharat Real-Time Pose Analyzer")
    parser.add_argument(
        "--sport", default="vertical_jump", choices=["vertical_jump", "snatch", "sprint", "javelin", "cricket_bat"]
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--athlete", default="athlete_01")
    args = parser.parse_args()

    run_analyzer(sport=args.sport, camera_id=args.camera, athlete_id=args.athlete)
