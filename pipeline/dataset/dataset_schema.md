# Personal Health — Biomechanical Training Dataset Schema

## Overview
Labeled dataset for training a sport-specific pose quality classifier.
Generated from sport-science literature distributions, calibrated against:
- **SportsPose** (CVPR 2023) — 176K 3D poses across 5 sports
- **AthletePose3D** (2025) — 1.3M frames, 12 competitive sport movements
- **MMPose CMJ Benchmark** — markerless vertical jump validated vs. force plates

## File: `training_data.csv`
Total: 2,000 labeled frames × 5 sports × 4 quality tiers

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| session_id | str | — | Unique session identifier |
| athlete_id | str | — | Athlete identifier (athlete_01..80) |
| frame_num | int | — | Frame number within session |
| sport | str | — | vertical_jump / snatch / sprint / javelin / cricket_bat |
| hip_angle_l | float | degrees | Left hip flexion angle (thigh-torso angle at hip) |
| hip_angle_r | float | degrees | Right hip flexion angle |
| knee_angle_l | float | degrees | Left knee flexion angle (thigh-shank angle) |
| knee_angle_r | float | degrees | Right knee flexion angle |
| shoulder_angle_l | float | degrees | Left shoulder abduction/elevation angle |
| shoulder_angle_r | float | degrees | Right shoulder angle |
| elbow_angle_l | float | degrees | Left elbow flexion angle |
| elbow_angle_r | float | degrees | Right elbow flexion angle |
| ankle_dorsiflexion_l | float | degrees | Left ankle dorsiflexion angle (knee-ankle-foot) |
| ankle_dorsiflexion_r | float | degrees | Right ankle dorsiflexion |
| trunk_lean | float | degrees | Forward trunk inclination from vertical |
| spine_deviation | float | degrees | Lateral spine deviation from mid-sagittal plane |
| shoulder_hip_sep | float | degrees | Rotational separation between shoulder and hip girdles |
| head_forward_pos | float | norm | Head anterior displacement relative to shoulders |
| com_height_norm | float | 0-1 | Centre of mass height normalized to body height |
| estimated_jump_height | float | cm | Estimated vertical jump height (0 for non-jump sports) |
| limb_symmetry_idx | float | 0-1 | Left-right limb symmetry (1.0 = perfect, < 0.85 = flag) |
| form_score | float | 0-100 | Biomechanical form quality score |
| phase_label | str | — | Movement phase: setup/descent/takeoff/flight/landing |
| quality_label | str | — | **TARGET LABEL**: elite / good / average / poor |
| feedback_tag | str | — | Primary coaching correction cue |

## Quality Label Definitions
| Label | Form Score | Description |
|-------|-----------|-------------|
| elite | 90-100 | Competition-ready biomechanics, near-optimal joint angles |
| good | 75-89 | Solid technique, minor deviations from ideal |
| average | 55-74 | Inconsistent mechanics, common beginner errors |
| poor | 0-54 | Significant form deviation, injury risk indicators |

## Key Biomechanical References
- **Vertical Jump**: Ideal descent knee angle ~90-105° (McMahon et al., 2018)
- **Snatch pull**: Hip angle at bar-lift ~85-95° (Stone et al., 2006)  
- **Sprint**: Hip drive angle ~45-60° at mid-stance (Novacheck, 1998)
- **Javelin release**: Elbow angle 100-150°, shoulder elevation ~165-180° (Best et al., 1993)
- **Symmetry threshold**: LSI < 0.85 = clinically significant asymmetry (Hooper et al., 2019)
