"""
=============================================================================
ActiveBharat — Biomechanical Pose Analyzer
=============================================================================
Uses MediaPipe BlazePose (33 keypoints) to extract real-time biomechanical
metrics from camera feed. Computes joint angles, CoM position, symmetry
index, trunk lean, and an overall form score for 5 sports:
  - vertical_jump (vj)
  - snatch (weightlifting)
  - sprint (20m)
  - javelin
  - cricket_bat

Key landmark indices (MediaPipe standard):
  0=Nose, 11=L_Shoulder, 12=R_Shoulder, 13=L_Elbow, 14=R_Elbow
  15=L_Wrist, 16=R_Wrist, 23=L_Hip, 24=R_Hip
  25=L_Knee, 26=R_Knee, 27=L_Ankle, 28=R_Ankle
=============================================================================
"""

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

try:
    from scipy.spatial.distance import mahalanobis
    from scipy.spatial.transform import Rotation as R
except ImportError:
    R = None
    mahalanobis = None

# ─── Data Types ─────────────────────────────────────────────────────────────


@dataclass
class Landmark:
    x: float  # normalized [0,1]
    y: float  # normalized [0,1]
    z: float  # depth estimate
    visibility: float  # confidence [0,1]


@dataclass
class BiomechanicalFrame:
    """All computed metrics for a single video frame."""

    # Joint angles (degrees)
    hip_angle_l: float = 0.0
    hip_angle_r: float = 0.0
    knee_angle_l: float = 0.0
    knee_angle_r: float = 0.0
    shoulder_angle_l: float = 0.0
    shoulder_angle_r: float = 0.0
    elbow_angle_l: float = 0.0
    elbow_angle_r: float = 0.0
    ankle_dorsiflexion_l: float = 0.0
    ankle_dorsiflexion_r: float = 0.0

    # Trunk & posture
    trunk_lean: float = 0.0  # forward lean from vertical (deg)
    spine_deviation: float = 0.0  # lateral deviation (deg)
    shoulder_hip_sep: float = 0.0  # rotation separation angle (deg)
    head_forward_pos: float = 0.0  # head anterior offset normalized

    # CoM & dynamics
    com_height_norm: float = 0.0  # normalized to body height [0,1]
    estimated_jump_height: float = 0.0  # cm

    # Symmetry
    limb_symmetry_idx: float = 1.0  # 1.0 = perfect symmetry

    # Composite scores
    form_score: float = 0.0  # 0-100 (smoothed via EMA, PF-06)
    raw_form_score: float = 0.0  # 0-100 unsmoothed (PF-06)
    form_quality: str = "unknown"  # elite/good/average/poor
    primary_feedback: str = ""  # top coaching cue

    # Phase 2 Kinematics
    phase_space_dm: float = 0.0  # Mahalanobis Distance for full trajectory
    torsion_error: float = 0.0  # Quaternion absolute rotational error (deg)
    dimensionless_jerk: float = 0.0  # Energy Efficiency Index (EEI)

    # Phase classification
    phase: str = "setup"  # setup/descent/takeoff/flight/landing

    # PF-07: Injury risk flags (asymmetry-based)
    injury_flags: list = field(default_factory=list)

    # Metadata
    visibility_ok: bool = True


# ─── Geometry Helpers ────────────────────────────────────────────────────────


def _angle_3pts(a: Landmark, b: Landmark, c: Landmark) -> float:
    """
    Compute the angle at vertex b using the three points a, b, c.
    Returns angle in degrees [0, 180].
    Math: θ = arccos( (BA · BC) / (|BA| |BC|) )
    """
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return 0.0
    cos_theta = np.clip(np.dot(ba, bc) / (norm_ba * norm_bc), -1.0, 1.0)
    return math.degrees(math.acos(cos_theta))


def _angle_vertical(p1: Landmark, p2: Landmark) -> float:
    """
    Angle of line segment p1→p2 relative to vertical axis.
    Returns angle in degrees (0 = perfectly vertical).
    """
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    if abs(dy) < 1e-6:
        return 90.0
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def _midpoint(p1: Landmark, p2: Landmark) -> Landmark:
    return Landmark(
        x=(p1.x + p2.x) / 2, y=(p1.y + p2.y) / 2, z=(p1.z + p2.z) / 2, visibility=min(p1.visibility, p2.visibility)
    )


def _distance(p1: Landmark, p2: Landmark) -> float:
    return math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)


# ─── Form Scoring Rules ──────────────────────────────────────────────────────

SPORT_IDEAL_ANGLES = {
    "general": {
        # Freeform training — wide acceptable ranges, focus on symmetry + posture
        "knee_angle": (60, 170),
        "hip_angle": (60, 170),
        "trunk_lean": (0, 30),
        "shoulder_angle": (20, 170),
        "elbow_angle": (30, 170),
        "symmetry": 0.85,
    },
    "vertical_jump": {
        # Countermovement Jump — descent phase ideal
        "knee_angle": (80, 110),  # deep squat at bottom
        "hip_angle": (80, 100),
        "trunk_lean": (0, 20),
        "ankle_dorsiflexion": (70, 110),
        "symmetry": 0.9,
    },
    "snatch": {
        "knee_angle": (90, 130),
        "hip_angle": (85, 110),
        "trunk_lean": (10, 35),
        "shoulder_angle": (30, 50),  # shoulder elevation
        "symmetry": 0.92,
    },
    "sprint": {
        "knee_angle": (80, 130),
        "hip_angle": (35, 70),  # hip drive
        "trunk_lean": (5, 20),
        "ankle_dorsiflexion": (60, 90),
        "symmetry": 0.85,
    },
    "javelin": {
        "shoulder_angle": (150, 180),  # throwing arm
        "elbow_angle": (100, 150),
        "trunk_lean": (20, 45),  # upper body rotation
        "shoulder_hip_sep": (30, 60),
        "symmetry": 0.75,
    },
    "cricket_bat": {
        "knee_angle": (110, 160),
        "hip_angle": (100, 140),
        "shoulder_angle": (60, 120),
        "trunk_lean": (10, 30),
        "symmetry": 0.80,
    },
    "squat": {
        "knee_angle": (75, 110),
        "hip_angle": (75, 105),
        "trunk_lean": (0, 25),
        "ankle_dorsiflexion": (65, 105),
        "symmetry": 0.92,
    },
    "push_up": {
        "elbow_angle": (85, 100),
        "shoulder_angle": (30, 60),
        "trunk_lean": (0, 8),
        "symmetry": 0.92,
    },
    "pull_up": {
        "elbow_angle": (30, 60),
        "shoulder_angle": (150, 180),
        "trunk_lean": (0, 15),
        "symmetry": 0.90,
    },
}


def compute_form_score(frame: BiomechanicalFrame, sport: str) -> tuple[float, str, str]:
    """
    Compute a 0-100 form score.

    Strategy: try the trained MLP classifier first (data flywheel path).
    If model prediction succeeds, use it as primary — it captures patterns
    the rule-based scorer can't. Fall through to rule-based scoring when
    the model isn't loaded (no TF, first deploy, inference error).

    Returns: (score, quality_label, primary_feedback)
    """
    # ── ML model path (primary when available) ──
    try:
        from services.model_registry import predict_quality

        frame_dict = {
            "hip_angle_l": frame.hip_angle_l,
            "hip_angle_r": frame.hip_angle_r,
            "knee_angle_l": frame.knee_angle_l,
            "knee_angle_r": frame.knee_angle_r,
            "shoulder_angle_l": frame.shoulder_angle_l,
            "shoulder_angle_r": frame.shoulder_angle_r,
            "elbow_angle_l": frame.elbow_angle_l,
            "elbow_angle_r": frame.elbow_angle_r,
            "ankle_dorsiflexion_l": frame.ankle_dorsiflexion_l,
            "ankle_dorsiflexion_r": frame.ankle_dorsiflexion_r,
            "trunk_lean": frame.trunk_lean,
            "spine_deviation": frame.spine_deviation,
            "shoulder_hip_sep": frame.shoulder_hip_sep,
            "head_forward_pos": frame.head_forward_pos,
            "com_height_norm": frame.com_height_norm,
            "estimated_jump_height": frame.estimated_jump_height,
            "limb_symmetry_idx": frame.limb_symmetry_idx,
        }
        prediction = predict_quality(frame_dict, sport)
        if prediction is not None and prediction["confidence"] >= 0.5:
            # Use model score but still generate rule-based feedback
            # (model gives quality class, rules give actionable coaching cue)
            _, _, rule_feedback = _rule_based_form_score(frame, sport)
            return prediction["form_score"], prediction["quality"], rule_feedback
    except Exception:
        pass  # fall through to rules

    # ── Rule-based fallback (always works, no dependencies) ──
    return _rule_based_form_score(frame, sport)


def _rule_based_form_score(frame: BiomechanicalFrame, sport: str) -> tuple[float, str, str]:
    """Rule-based form scoring from sport-specific ideal angle ranges."""
    if sport not in SPORT_IDEAL_ANGLES:
        sport = "vertical_jump"  # default

    rules = SPORT_IDEAL_ANGLES[sport]
    score_components = []
    violations = []

    def check_range(value, ideal_range, weight=1.0, label=""):
        lo, hi = ideal_range
        if lo <= value <= hi:
            score_components.append(100.0 * weight)
        else:
            # penalty is proportional to deviation
            deviation = min(abs(value - lo), abs(value - hi))
            penalty = min(deviation / 20.0, 1.0)  # max 20deg deviation = full penalty
            score_components.append(max(0, (1.0 - penalty) * 100.0) * weight)
            if penalty > 0.3:
                violations.append(label)

    # Apply sport-specific checks
    if "knee_angle" in rules:
        avg_knee = (frame.knee_angle_l + frame.knee_angle_r) / 2
        check_range(avg_knee, rules["knee_angle"], weight=2.0, label="Adjust Knee Bend")

    if "hip_angle" in rules:
        avg_hip = (frame.hip_angle_l + frame.hip_angle_r) / 2
        check_range(avg_hip, rules["hip_angle"], weight=2.0, label="Lower Your Hips")

    if "trunk_lean" in rules:
        check_range(frame.trunk_lean, rules["trunk_lean"], weight=1.5, label="Control Trunk Lean")

    if "shoulder_angle" in rules:
        avg_shoulder = (frame.shoulder_angle_l + frame.shoulder_angle_r) / 2
        check_range(avg_shoulder, rules["shoulder_angle"], weight=1.0, label="Adjust Arm Position")

    if "elbow_angle" in rules:
        avg_elbow = (frame.elbow_angle_l + frame.elbow_angle_r) / 2
        check_range(avg_elbow, rules["elbow_angle"], weight=1.0, label="Adjust Elbow Angle")

    if "ankle_dorsiflexion" in rules:
        avg_ankle = (frame.ankle_dorsiflexion_l + frame.ankle_dorsiflexion_r) / 2
        check_range(avg_ankle, rules["ankle_dorsiflexion"], weight=1.0, label="Control Ankle Flex")

    if "shoulder_hip_sep" in rules:
        check_range(
            frame.shoulder_hip_sep, rules["shoulder_hip_sep"], weight=1.0, label="Improve Hip-Shoulder Separation"
        )

    # Symmetry penalty
    sym_min = rules.get("symmetry", 0.85)
    if frame.limb_symmetry_idx < sym_min:
        sym_loss = (sym_min - frame.limb_symmetry_idx) * 100
        score_components.append(max(0, 100 - sym_loss * 2))
        if sym_loss > 10:
            violations.append("Balance Left-Right Movement")
    else:
        score_components.append(100.0)

    total = sum(score_components) / len(score_components) if score_components else 50.0
    score = round(min(100, max(0, total)), 1)

    if score >= 90:
        quality = "elite"
    elif score >= 75:
        quality = "good"
    elif score >= 55:
        quality = "average"
    else:
        quality = "poor"

    feedback = (
        violations[0] if violations else ("Great form! Maintain position." if score >= 80 else "Keep practicing!")
    )
    return score, quality, feedback


# ─── Phase-Space Kinematics ────────────────────────────────────────────────


def _compute_quaternion_torsion(a: Landmark, b: Landmark, c: Landmark) -> float:
    """Returns absolute 3D rotational torsion error using Quaternions."""
    if not R:
        return 0.0

    v1 = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    v2 = np.array([c.x - b.x, c.y - b.y, c.z - b.z])

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0

    v1 = v1 / n1
    v2 = v2 / n2

    cross = np.cross(v1, v2)
    norm_cross = np.linalg.norm(cross)
    if norm_cross < 1e-6:
        return 0.0

    # Create the rotation quaternion for the joint
    dot = np.dot(v1, v2)
    q_user = R.from_quat([cross[0], cross[1], cross[2], 1 + dot])

    # In a real environment, we compare q_user to q_pro from a DB
    # For now, we measure the raw magnitude of twist vs planar stability
    theta = 2 * math.acos(min(1.0, abs(q_user.as_quat()[3])))
    return math.degrees(theta)


def _compute_dimensionless_jerk(com_trajectory: list) -> float:
    """Calculates Energy Efficiency Index (EEI) over a trajectory window."""
    if len(com_trajectory) < 5:
        return 0.0

    y = np.array(com_trajectory)
    dt = 1.0 / 30.0  # assuming 30 fps

    # 1st deriv: velocity, 2nd: accel, 3rd: jerk
    v = np.gradient(y, dt)
    a = np.gradient(v, dt)
    j = np.gradient(a, dt)

    integral_j2 = np.trapz(j**2, dx=dt)

    duration = len(com_trajectory) * dt
    path_length = np.sum(np.abs(np.diff(y)))
    if path_length < 1e-6:
        return 0.0

    # DJ = (T^5 / L^2) * integral(J^2)
    dj = (duration**5 / path_length**2) * integral_j2
    return abs(float(dj))


_SPORT_PHASE_IDEALS = {
    "vertical_jump": (90.0, 0.0),
    "squat": (85.0, 0.0),
    "snatch": (100.0, 0.0),
    "sprint": (90.0, 50.0),
    "push_up": (90.0, 0.0),
    "pull_up": (45.0, 0.0),
    "javelin": (150.0, 0.0),
    "cricket_bat": (135.0, 0.0),
}


def _compute_mahalanobis_phase_space(
    current_angle: float, past_angle: float, dt: float, sport: str = "vertical_jump"
) -> float:
    """Calculates 1D Phase-Space Manifold Mahalanobis Distance."""
    if not mahalanobis:
        return 0.0

    velocity = (current_angle - past_angle) / dt
    state = np.array([current_angle, velocity])

    ideal_angle, ideal_velocity = _SPORT_PHASE_IDEALS.get(sport, (90.0, 0.0))
    mu_pro = np.array([ideal_angle, ideal_velocity])
    cov_inv = np.array([[0.1, 0.0], [0.0, 0.5]])

    dist = mahalanobis(state, mu_pro, cov_inv)
    return float(dist)


# ─── Phase Classification ────────────────────────────────────────────────────────


def classify_jump_phase(frames_history) -> str:
    """Simple heuristic to classify the athletic phase (e.g. vertical jump)."""
    if len(frames_history) < 5:
        return "setup"

    # frames_history is an iterable (deque or list)
    recent = list(frames_history)[-5:]
    recent_com = [f.com_height_norm for f in recent]
    avg_recent = sum(recent_com) / len(recent_com)
    delta = recent_com[-1] - recent_com[0]
    avg_knee = (frames_history[-1].knee_angle_l + frames_history[-1].knee_angle_r) / 2

    if avg_knee > 150:  # legs nearly straight
        if avg_recent > 0.6:
            return "setup"
        elif delta < -0.02:
            return "landing"
        else:
            return "flight"
    elif delta < -0.02:  # CoM dropping
        return "descent"
    elif delta > 0.05:  # CoM rising fast
        return "takeoff"
    else:
        return "setup"


# PF-05: Phase detection for push_up, pull_up
def _classify_rep_phase(frames_history, sport: str) -> str:
    if len(frames_history) < 5:
        return "ready"
    recent = list(frames_history)[-5:]
    avg_elbow = sum((f.elbow_angle_l + f.elbow_angle_r) / 2 for f in recent) / len(recent)
    delta_elbow = recent[-1].elbow_angle_l - recent[0].elbow_angle_l
    if sport == "push_up":
        if avg_elbow > 150:
            return "up"
        elif delta_elbow < -3:
            return "descent"
        elif avg_elbow < 100:
            return "bottom"
        else:
            return "ascent"
    else:  # pull_up
        if avg_elbow > 150:
            return "hang"
        elif delta_elbow < -3:
            return "pull"
        elif avg_elbow < 90:
            return "top"
        else:
            return "descent"


# PF-05: Phase detection for sprint
def _classify_sprint_phase(frames_history) -> str:
    if len(frames_history) < 5:
        return "start"
    recent = list(frames_history)[-5:]
    avg_trunk = sum(f.trunk_lean for f in recent) / len(recent)
    delta_com = recent[-1].com_height_norm - recent[0].com_height_norm
    if avg_trunk > 30:
        return "start"
    elif delta_com > 0.02:
        return "acceleration"
    elif avg_trunk < 10:
        return "max_velocity"
    else:
        return "deceleration"


# PF-05: Phase detection for javelin, cricket_bat
def _classify_throw_phase(frames_history, sport: str) -> str:
    if len(frames_history) < 5:
        return "stance"
    recent = list(frames_history)[-5:]
    avg_shoulder = sum((f.shoulder_angle_l + f.shoulder_angle_r) / 2 for f in recent) / len(recent)
    delta_shoulder = recent[-1].shoulder_angle_l - recent[0].shoulder_angle_l
    if sport == "javelin":
        if delta_shoulder > 5:
            return "approach"
        elif avg_shoulder > 140:
            return "crossover"
        elif delta_shoulder < -10:
            return "delivery"
        else:
            return "follow_through"
    else:  # cricket_bat
        if avg_shoulder < 60:
            return "stance"
        elif delta_shoulder > 5:
            return "backswing"
        elif delta_shoulder < -5:
            return "downswing"
        else:
            return "follow_through"


# PF-05: Generic fallback for unknown sports
def _classify_generic_phase(frames_history) -> str:
    if len(frames_history) < 5:
        return "ready"
    recent = list(frames_history)[-5:]
    delta_com = recent[-1].com_height_norm - recent[0].com_height_norm
    if abs(delta_com) < 0.01:
        return "ready"
    elif delta_com > 0:
        return "active"
    else:
        return "recovery"


# ─── Main Analyzer ──────────────────────────────────────────────────────────


class PoseAnalyzer:
    """
    Main orchestrator: converts raw MediaPipe landmarks into a
    BiomechanicalFrame with all derived metrics.
    """

    MIN_VISIBILITY = 0.3  # lowered from 0.5 — phone camera poses often have partial occlusion

    # Smoothing buffer for temporal consistency
    SMOOTH_WINDOW = 5

    def __init__(self, sport: str = "vertical_jump", body_height_cm: float = 170):
        self.sport = sport
        self.body_height_cm = body_height_cm  # PF-03: athlete's actual height, no longer hardcoded
        self.frame_history: deque = deque(maxlen=60)  # Sliding Window for derivatives
        self._baseline_body_height = None
        self._baseline_com_y = None
        self._com_history: deque = deque(maxlen=60)
        self._mp_pose = None  # Reusable MediaPipe Pose instance
        self._smoothed_score = None  # PF-06: EMA temporal smoothing

    def set_sport(self, sport: str):
        self.sport = sport
        self.frame_history.clear()
        self._baseline_body_height = None
        self._baseline_com_y = None
        self._com_history.clear()
        self._smoothed_score = None  # PF-06: reset EMA on sport change

    def _to_landmark(self, lm) -> Landmark:
        """Convert MediaPipe NormalizedLandmark to our Landmark type."""
        return Landmark(
            x=lm.x, y=lm.y, z=lm.z,
            visibility=getattr(lm, "visibility", getattr(lm, "presence", 0.9)),
        )

    def _is_visible(self, *landmarks: Landmark) -> bool:
        return all(l.visibility >= self.MIN_VISIBILITY for l in landmarks)

    def _calibrate(self, lms: list[Landmark]):
        """Set body-height baseline from first good frame."""
        nose, l_ankle = lms[0], lms[27]
        if self._is_visible(nose, l_ankle):
            h = abs(nose.y - l_ankle.y)
            if h > 0.1:
                self._baseline_body_height = h
                l_hip, r_hip = lms[23], lms[24]
                if self._is_visible(l_hip, r_hip):
                    self._baseline_com_y = (l_hip.y + r_hip.y) / 2

    def analyze(self, mediapipe_results) -> BiomechanicalFrame:
        """
        Primary entry point: pass in mediapipe pose_landmarks.
        Returns a fully populated BiomechanicalFrame.
        """
        frame = BiomechanicalFrame()

        if not mediapipe_results or not mediapipe_results.pose_landmarks:
            frame.visibility_ok = False
            return frame

        lms_raw = mediapipe_results.pose_landmarks.landmark
        lms = [self._to_landmark(l) for l in lms_raw]

        # Calibrate body height on first frame
        if self._baseline_body_height is None:
            self._calibrate(lms)

        # Extract named landmarks
        nose = lms[0]
        l_sh, r_sh = lms[11], lms[12]
        l_el, r_el = lms[13], lms[14]
        l_wr, r_wr = lms[15], lms[16]
        l_hip, r_hip = lms[23], lms[24]
        l_kn, r_kn = lms[25], lms[26]
        l_an, r_an = lms[27], lms[28]
        l_foot, r_foot = lms[31], lms[32]

        # Check visibility — require at least one hip + one knee visible
        # (real phone camera often has partial occlusion on one side)
        hip_ok = l_hip.visibility >= self.MIN_VISIBILITY or r_hip.visibility >= self.MIN_VISIBILITY
        knee_ok = l_kn.visibility >= self.MIN_VISIBILITY or r_kn.visibility >= self.MIN_VISIBILITY
        frame.visibility_ok = hip_ok and knee_ok
        if not frame.visibility_ok:
            return frame

        # ── Joint Angles ──────────────────────────────
        # Hip angles (torso-hip-knee)
        if self._is_visible(l_sh, l_hip, l_kn):
            frame.hip_angle_l = _angle_3pts(l_sh, l_hip, l_kn)
        if self._is_visible(r_sh, r_hip, r_kn):
            frame.hip_angle_r = _angle_3pts(r_sh, r_hip, r_kn)

        # Knee angles (hip-knee-ankle)
        if self._is_visible(l_hip, l_kn, l_an):
            frame.knee_angle_l = _angle_3pts(l_hip, l_kn, l_an)
        if self._is_visible(r_hip, r_kn, r_an):
            frame.knee_angle_r = _angle_3pts(r_hip, r_kn, r_an)

        # Shoulder angles (elbow-shoulder-hip)
        if self._is_visible(l_el, l_sh, l_hip):
            frame.shoulder_angle_l = _angle_3pts(l_el, l_sh, l_hip)
        if self._is_visible(r_el, r_sh, r_hip):
            frame.shoulder_angle_r = _angle_3pts(r_el, r_sh, r_hip)

        # Elbow angles (shoulder-elbow-wrist)
        if self._is_visible(l_sh, l_el, l_wr):
            frame.elbow_angle_l = _angle_3pts(l_sh, l_el, l_wr)
        if self._is_visible(r_sh, r_el, r_wr):
            frame.elbow_angle_r = _angle_3pts(r_sh, r_el, r_wr)

        # Ankle dorsiflexion (knee-ankle-foot)
        if self._is_visible(l_kn, l_an, l_foot):
            frame.ankle_dorsiflexion_l = _angle_3pts(l_kn, l_an, l_foot)
        if self._is_visible(r_kn, r_an, r_foot):
            frame.ankle_dorsiflexion_r = _angle_3pts(r_kn, r_an, r_foot)

        # ── Trunk & Posture ───────────────────────────
        mid_sh = _midpoint(l_sh, r_sh)
        mid_hip = _midpoint(l_hip, r_hip)
        frame.trunk_lean = _angle_vertical(mid_hip, mid_sh)

        # Spine lateral deviation (angle of lateral offset from vertical)
        lateral_offset = abs(mid_sh.x - mid_hip.x)
        vertical_dist = abs(mid_sh.y - mid_hip.y) + 1e-6
        frame.spine_deviation = math.degrees(math.atan2(lateral_offset, vertical_dist))

        # Hip-shoulder rotation separation (angle between shoulder line and hip line)
        sh_dx = l_sh.x - r_sh.x
        sh_dy = l_sh.z - r_sh.z
        hip_dx = l_hip.x - r_hip.x
        hip_dy = l_hip.z - r_hip.z
        sh_angle = math.atan2(sh_dy, sh_dx + 1e-9)
        hip_angle = math.atan2(hip_dy, hip_dx + 1e-9)
        frame.shoulder_hip_sep = abs(math.degrees(sh_angle - hip_angle))

        # Head forward position
        if self._is_visible(nose, mid_sh):
            frame.head_forward_pos = (mid_sh.x - nose.x) * 50  # normalized units

        # ── CoM & dynamics ────────────────────────────
        com_y = mid_hip.y  # Approximate CoM
        self._com_history.append(com_y)
        # Deque automatically handles maxlen=60, no manual pop needed.

        if self._baseline_body_height and self._baseline_com_y:
            com_displacement = self._baseline_com_y - com_y  # positive = rising
            frame.com_height_norm = min(1.0, max(0.0, 0.5 + com_displacement / self._baseline_body_height))
            # Estimate jump height: h = com_displacement * body_height_in_cm / body_height_norm
            # Average body height ~170cm for Indian male athletes
            if com_displacement > 0:
                frame.estimated_jump_height = (com_displacement / self._baseline_body_height) * self.body_height_cm
        else:
            frame.com_height_norm = 0.5

        # ── Symmetry Index ────────────────────────────
        # Compare left/right metrics: lower deviation = better symmetry
        sym_pairs = [
            (frame.hip_angle_l, frame.hip_angle_r),
            (frame.knee_angle_l, frame.knee_angle_r),
            (frame.ankle_dorsiflexion_l, frame.ankle_dorsiflexion_r),
        ]
        valid_pairs = [(a, b) for a, b in sym_pairs if a > 0 and b > 0]
        if valid_pairs:
            avg_asymmetry = sum(abs(a - b) / max(a, b, 1) for a, b in valid_pairs) / len(valid_pairs)
            frame.limb_symmetry_idx = round(max(0.0, 1.0 - avg_asymmetry), 3)

        # PF-07: Injury risk flags based on asymmetry
        if frame.limb_symmetry_idx < 0.70:
            frame.injury_flags.append("asymmetry_critical")
        elif frame.limb_symmetry_idx < 0.80:
            # Check if last 3 frames also had low symmetry
            recent = list(self.frame_history)[-3:]
            if len(recent) >= 2 and all(f.limb_symmetry_idx < 0.80 for f in recent):
                frame.injury_flags.append("asymmetry_warning")

        # ── Phase Classification (PF-05: all 8 sports) ──
        if self.sport in ("vertical_jump", "squat", "snatch"):
            frame.phase = classify_jump_phase(self.frame_history)
        elif self.sport in ("push_up", "pull_up"):
            frame.phase = _classify_rep_phase(self.frame_history, self.sport)
        elif self.sport == "sprint":
            frame.phase = _classify_sprint_phase(self.frame_history)
        elif self.sport in ("javelin", "cricket_bat"):
            frame.phase = _classify_throw_phase(self.frame_history, self.sport)
        else:
            frame.phase = _classify_generic_phase(self.frame_history)

        # ── Phase 2: Advanced Kinematics (math2.pdf) ──
        # 1. 3D Torsion (Quaternions)
        if self._is_visible(l_hip, l_kn, l_an):
            frame.torsion_error = _compute_quaternion_torsion(l_hip, l_kn, l_an)

        # 2. Phase-Space Manifold (Mahalanobis)
        if len(self.frame_history) >= 2:
            past_frame = self.frame_history[-2]
            dt = 1.0 / 30.0  # assuming ~30fps
            frame.phase_space_dm = _compute_mahalanobis_phase_space(
                frame.knee_angle_l, past_frame.knee_angle_l, dt, self.sport
            )

        # 3. Dimensionless Jerk (EEI)
        if len(self._com_history) > 10:
            frame.dimensionless_jerk = _compute_dimensionless_jerk(list(self._com_history))

        # ── Form Score ────────────────────────────────
        raw_score, frame.form_quality, frame.primary_feedback = compute_form_score(frame, self.sport)
        frame.raw_form_score = raw_score
        # PF-06: EMA temporal smoothing (alpha=0.3) — single bad frame can't tank the score
        if self._smoothed_score is None:
            self._smoothed_score = raw_score
        else:
            self._smoothed_score = 0.3 * raw_score + 0.7 * self._smoothed_score
        frame.form_score = round(self._smoothed_score, 1)

        # Store to history (deque auto-truncates)
        self.frame_history.append(frame)

        return frame

    def analyze_base64_image(self, image_b64: str, sport: str = None) -> dict:
        """
        Analyze a single frame from a base64-encoded JPEG (from mobile camera).
        Returns a metrics dict or {'pose_detected': False} if no person found.
        Called by: POST /session/{id}/frame when image_b64 field is present.

        Uses MediaPipe PoseLandmarker Task API (0.10.x+). Falls back to legacy
        solutions API for older installs.
        """
        import base64
        import types
        from pathlib import Path

        import numpy as np

        try:
            import cv2
            import mediapipe as mp
        except ImportError:
            return {"pose_detected": False, "error": "cv2/mediapipe not installed on server"}

        prev_sport = self.sport
        if sport:
            self.sport = sport
        try:
            raw = base64.b64decode(image_b64)
            arr = np.frombuffer(raw, np.uint8)
            frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                return {"pose_detected": False, "error": "image decode failed"}

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            # ── Try new Task API first (MediaPipe 0.10.x+) ──
            landmarks_list = None
            try:
                from mediapipe.tasks.python import BaseOptions
                from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

                if self._mp_pose is None or not isinstance(self._mp_pose, PoseLandmarker):
                    model_path = Path(__file__).parent.parent / "models" / "pose_landmarker.task"
                    if not model_path.exists():
                        raise FileNotFoundError(f"Model not found: {model_path}")
                    options = PoseLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=str(model_path)),
                        running_mode=RunningMode.IMAGE,
                        num_poses=1,
                        min_pose_detection_confidence=0.5,
                        min_tracking_confidence=0.5,
                    )
                    self._mp_pose = PoseLandmarker.create_from_options(options)

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                detection = self._mp_pose.detect(mp_image)

                if detection.pose_landmarks and len(detection.pose_landmarks) > 0:
                    landmarks_list = detection.pose_landmarks[0]
            except (ImportError, FileNotFoundError):
                # Fall back to legacy solutions API
                try:
                    mp_pose = mp.solutions.pose
                    if self._mp_pose is None:
                        self._mp_pose = mp_pose.Pose(
                            static_image_mode=True,
                            model_complexity=1,
                            min_detection_confidence=0.5,
                        )
                    results = self._mp_pose.process(rgb)
                    if results.pose_landmarks:
                        landmarks_list = results.pose_landmarks.landmark
                except AttributeError:
                    return {"pose_detected": False, "error": "mediapipe API unavailable"}

            if not landmarks_list or len(landmarks_list) < 33:
                return {"pose_detected": False}

            # Convert to the format self.analyze() expects:
            # a namespace with .pose_landmarks.landmark = list of objects with .x,.y,.z,.visibility
            lms = [
                types.SimpleNamespace(
                    x=getattr(lm, "x", 0),
                    y=getattr(lm, "y", 0),
                    z=getattr(lm, "z", 0),
                    visibility=getattr(lm, "visibility", getattr(lm, "presence", 0.9)),
                )
                for lm in landmarks_list
            ]
            mock_results = types.SimpleNamespace(
                pose_landmarks=types.SimpleNamespace(landmark=lms)
            )

            # Multi-person guard
            multi_person_warning = None
            avg_visibility = sum(l.visibility for l in lms) / len(lms)
            if avg_visibility < 0.6:
                multi_person_warning = "Low visibility — ensure only you are in frame"

            bio = self.analyze(mock_results)
            result = {
                "pose_detected": True,
                "visibility_ok": bio.visibility_ok,
                "form_score": bio.form_score,
                "form_quality": bio.form_quality,
                "primary_feedback": bio.primary_feedback,
                "phase": bio.phase,
                "joint_angles": {
                    "KNEE_L": round(bio.knee_angle_l, 1),
                    "KNEE_R": round(bio.knee_angle_r, 1),
                    "HIP_L": round(bio.hip_angle_l, 1),
                    "HIP_R": round(bio.hip_angle_r, 1),
                    "TRUNK": round(bio.trunk_lean, 1),
                    "ELBOW_L": round(bio.elbow_angle_l, 1),
                    "ELBOW_R": round(bio.elbow_angle_r, 1),
                    "SHOULDER_L": round(bio.shoulder_angle_l, 1),
                    "SHOULDER_R": round(bio.shoulder_angle_r, 1),
                    "ANKLE_L": round(bio.ankle_dorsiflexion_l, 1),
                    "ANKLE_R": round(bio.ankle_dorsiflexion_r, 1),
                },
                "symmetry_score": round(bio.limb_symmetry_idx, 3),
                "trunk_lean": round(bio.trunk_lean, 1),
                "estimated_jump_height": round(bio.estimated_jump_height, 1),
                "com_height_norm": round(bio.com_height_norm, 3),
                "spine_deviation": round(bio.spine_deviation, 1),
                "shoulder_hip_sep": round(bio.shoulder_hip_sep, 1),
                "head_forward_pos": round(bio.head_forward_pos, 2),
                "injury_flags": bio.injury_flags,
            }
            if multi_person_warning:
                result["warning"] = multi_person_warning
            return result
        finally:
            self.sport = prev_sport

    def get_session_summary(self, athlete_id: str, session_id: str) -> dict:
        """Aggregate all frames in session into a summary dict."""
        if not self.frame_history:
            return {}

        scores = [f.form_score for f in self.frame_history if f.visibility_ok]
        jump_heights = [f.estimated_jump_height for f in self.frame_history if f.estimated_jump_height > 0]
        symmetries = [f.limb_symmetry_idx for f in self.frame_history if f.visibility_ok]

        return {
            "session_id": session_id,
            "athlete_id": athlete_id,
            "sport": self.sport,
            "total_frames": len(self.frame_history),
            "valid_frames": len(scores),
            "avg_form_score": round(sum(scores) / len(scores), 1) if scores else 0,
            "peak_form_score": round(max(scores), 1) if scores else 0,
            "peak_jump_height_cm": round(max(jump_heights), 1) if jump_heights else 0,
            "avg_jump_height_cm": round(sum(jump_heights) / len(jump_heights), 1) if jump_heights else 0,
            "avg_symmetry": round(sum(symmetries) / len(symmetries), 2) if symmetries else 0,
            "quality_distribution": {
                "elite": sum(1 for f in self.frame_history if f.form_quality == "elite"),
                "good": sum(1 for f in self.frame_history if f.form_quality == "good"),
                "average": sum(1 for f in self.frame_history if f.form_quality == "average"),
                "poor": sum(1 for f in self.frame_history if f.form_quality == "poor"),
            },
            "key_metrics": {
                "avg_knee_angle": round(
                    sum((f.knee_angle_l + f.knee_angle_r) / 2 for f in self.frame_history if f.visibility_ok)
                    / max(len(scores), 1),
                    1,
                ),
                "avg_hip_angle": round(
                    sum((f.hip_angle_l + f.hip_angle_r) / 2 for f in self.frame_history if f.visibility_ok)
                    / max(len(scores), 1),
                    1,
                ),
                "avg_trunk_lean": round(
                    sum(f.trunk_lean for f in self.frame_history if f.visibility_ok) / max(len(scores), 1), 1
                ),
            },
        }
