# Motioncode Dimensional Space — Summary for Visualization

## What This Is
A 208-dimensional discrete coordinate system where each human motion clip maps to a single point. Built by analyzing 3D joint coordinates through rule-based posecode→motioncode pipeline (from MotionScript repo). No neural networks or LLMs involved — purely geometric analysis with hand-tuned thresholds.

## Data Files
- `motioncode_output/vectors.npz` — contains:
  - `vectors`: numpy array, shape `(5532, 208)`, dtype int32. Each row is one motion.
  - `motion_ids`: numpy array of strings, shape `(5532,)`. Filename (without .npy) for each row.
- `motioncode_output/metadata.json` — contains:
  - `labels`: list of 208 strings, one per dimension (e.g., `"left_hip+left_knee+left_ankle|angular|intensity"`)
  - `categories`: dict mapping each motioncode kind to its intensity category names and thresholds, plus shared velocity categories
- `verb_groups copy.json` — 46 motion groups (walk, jump, run, etc.) with lists of motion IDs per group
- `verb_group_centers.json` — centroid file and top-3 nearest files per group
- `space_analysis.json` — per-group and global statistics (unique points, volume, mean/std)

## Loading
```python
import numpy as np, json
data = np.load('motioncode_output/vectors.npz', allow_pickle=True)
vectors = data['vectors']       # (5532, 208) int32
motion_ids = data['motion_ids'] # (5532,) strings
with open('motioncode_output/metadata.json') as f:
    meta = json.load(f)
labels = meta['labels']         # 208 dimension labels
```

## Dimension Structure
208 dimensions total. Each dimension is one of:
- **intensity** (how much change) — integer, range varies by category kind
- **velocity** (how fast) — integer, range 0–4 for all kinds

Value of **-1** means no motion detected for that slot.

### The 208 dimensions break down as:
- **110 base dimensions** (55 unique joint-set × motioncode-kind slots, each with intensity + velocity)
- **98 Option B duplicates** (pairwise slots stored under both joints)

### 11 Motioncode Kinds

| Kind | What it measures | Joint sets | Intensity range | Intensity categories |
|---|---|---|---|---|
| angular | Joint bend/extend | 4 (L/R knee, L/R elbow) | 0–6 | significant_bend, moderate_bend, slight_bend, no_action, slight_extension, moderate_extension, significant_extension |
| proximity | Two joints closing/spreading | 22 pairs | 0–4 | significant_closing, moderate_closing, stationary, moderate_spreading, significant_spreading |
| spatial_relation_x | Relative left/right shift | 5 pairs | 0–2 | left-to-right, stationary, right-to-left |
| spatial_relation_y | Relative up/down shift | 13 pairs | 0–2 | above-to-below, stationary, below-to-above |
| spatial_relation_z | Relative front/back shift | 5 pairs | 0–2 | front-to-behind, stationary, behind-to-front |
| displacement_x | Body lateral movement | 1 (translation) | 0–10 | very_long_left → no_action → very_long_right |
| displacement_y | Body vertical movement | 1 (translation) | 0–10 | very_long_down → no_action → very_long_up |
| displacement_z | Body forward/back movement | 1 (translation) | 0–10 | very_long_backward → no_action → very_long_forward |
| rotation_pitch | Body lean forward/back | 1 (orientation) | 0–6 | significant_leaning_backward → no_action → significant_leaning_forward |
| rotation_roll | Body tilt left/right | 1 (orientation) | 0–6 | significant_leaning_right → no_action → significant_leaning_left |
| rotation_yaw | Body turn clockwise/counter | 1 (orientation) | 0–6 | significant_turn_clockwise → no_action → significant_turn_counterclockwise |

### Shared Velocity Categories (all kinds)
| Index | Name | Threshold |
|---|---|---|
| 0 | very_slow | ≤0.05 |
| 1 | slow | ≤0.10 |
| 2 | moderate | ≤0.50 |
| 3 | fast | ≤0.80 |
| 4 | very_fast | >0.80 |

## Key Properties
- **Discrete**: All values are integers. -1 = inactive, 0+ = category index.
- **Sparse**: Most motions activate 30–100 of 208 dimensions. Rest are -1.
- **Deterministic**: Same input always produces same output (no randomization).
- **Independent per motion**: No cross-motion dependencies. Each motion's vector depends only on its own joint coordinates.

## Global Statistics (5532 motions)
- Active dimensions: 204 / 208
- Unique points: 5011 (91% of motions are unique)
- Mean coordinate (excluding -1): 0.26
- Std coordinate: 2.13
- Average range per dimension: 10.06

## Group Statistics (41 groups)
- Largest: walk (2663 motions, 2246 unique points)
- Most spread: dance (204 active dims, 97.9% log volume coverage)
- Most compact: stomp (50 active dims, 20.2% log volume coverage)
- Tightest cluster: turn (mean distance to centroid: 19.18)
- Loosest cluster: fall (mean distance to centroid: 53.58)

## Dimension Labels Format
Each label follows: `joint(s)|motioncode_kind|value_type`
- Single joint: `"left_hip+left_knee+left_ankle|angular|intensity"`
- Joint pair: `"left_hand+right_hand|proximity|velocity"`
- Duplicate (Option B): `"right_hand+left_hand|proximity|intensity(dup)"`
- Global: `"translation|displacement_z|intensity"`, `"orientation|rotation_yaw|velocity"`

## Decoding a Value
To convert integer index to human-readable name:
```python
# For intensity: look up the motioncode kind from the label
kind = label.split('|')[1]  # e.g., "angular"
categories = meta['categories'][kind]['intensity_categories']
name = categories[value]  # e.g., categories[0] = "significant_bend"

# For velocity (same for all kinds):
vel_categories = meta['categories']['_velocity']['categories']
name = vel_categories[value]  # e.g., vel_categories[3] = "fast"
```

## Equation
```
D = Σ(solo kinds)[joint_sets × 2] + Σ(pairwise kinds)[joint_sets × 2 × 2] = 208
Solo:     (4 + 1 + 1 + 1 + 1 + 1 + 1) × 2 = 20
Pairwise: (22 + 5 + 13 + 5) × 2 × 2 = 180
Extra from angular triplet duplication: 4 × 2 = 8
Total: 208
```

## Scripts

| Script | Purpose |
|---|---|
| `extract_motioncodes.py` | Core pipeline. Takes joint coordinates → posecodes → motioncodes → 208-d vector. Contains all operator definitions, thresholds, joint sets. Also has `load_humanml3d()` to load from `new_joints/` + `new_joint_vecs/`, `filter_motions_by_lower_body()` to filter motions with lower-body activity, `score_lower_body()` to produce per-motion lower-body scoring JSON, `process_motion_files()` + `save_space()` + `load_space()` for batch processing and saving the (N, 208) matrix. Edit the data definition lists at the top to add/remove joints or categories. |
| `plot_verb_groups.py` | Computes centroid of each verb group in the 208-d space, finds the nearest file(s) to each centroid. Outputs `verb_group_centers.json`. |
| `space_analysis.py` | Data analysis of the dimensional space. Computes per-group and global stats: mean, std, active dimensions, unique points, log continuous volume, log discrete volume, cumulative relative frequency of space coverage. Outputs `space_analysis.json`. |
| `motioncode_space.json` | Reference JSON with all variable definitions: every operator's thresholds, category names, joint sets, and the dimension summary. Not a script — a data file for lookup. |
| `dimension_diagram.md` | ASCII diagram of the dimension equation showing how solo kinds, pairwise kinds, intensity, and velocity contribute to the 208 total. |
