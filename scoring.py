import json
import numpy as np
import matplotlib.pyplot as plt

#two ways to go about it: normalize categories by max value (Current) or normalize each feature column
#across motions (Commented). Similar results but the current gave away hand-related motion 
# very well. Commented may be used for getting better leg movement from up.

#normal score is better (again, maybe just for removing) than normalized (Commented), which makes sure that there is no difference between having higher and lower number of params

def get_feature_max(feature):
    body, relation, metric = feature.split("|")

    if relation == "angular":
        if metric == "intensity":
            return 6
        elif metric == "velocity":
            return 4

    elif relation == "proximity":
        return 4

    elif relation in ["spatial_relation_x", "spatial_relation_y"]:
        if metric == "intensity":
            return 2
        elif metric == "velocity":
            return 4

    return 1  # fallback

# ---------- LOAD ----------
with open("lower_body_scores.json", "r") as f:
    data = json.load(f)

motion_ids = list(data.keys())

# ---------- DEFINE FEATURE SPACE ----------
ALL_FEATURES = set()
for m in motion_ids:
    ALL_FEATURES.update(data[m]["lower_body_scores"].keys())
ALL_FEATURES = sorted(list(ALL_FEATURES))

print("num features:", len(ALL_FEATURES))

# ---------- REBUILD FULL MATRIX ----------
X = []
for m in motion_ids:
    sparse = data[m]["lower_body_scores"]
    row = [sparse.get(f, -1) for f in ALL_FEATURES]
    X.append(row)

X = np.array(X, dtype=float)   # shape: (motions, features)

# =========================================================
# PRESTEP: NORMALIZE EACH FEATURE ACROSS MOTIONS
# =========================================================
# -1 means inactive, so ignore those when normalizing.
# Each feature column is normalized only over its active values.

X_norm = X.copy()

for j, feature in enumerate(ALL_FEATURES):
    max_val = get_feature_max(feature)

    if max_val == 0:
        continue

    col = X[:, j]
    active_mask = (col != -1)

    # normalize only active values
    X_norm[active_mask, j] = col[active_mask] / max_val

#replace from line 60
"""
for j in range(X.shape[1]):
    col = X[:, j]
    active_mask = (col != -1)

    if np.sum(active_mask) == 0:
        continue

    active_vals = col[active_mask]

    # min-max normalization within this feature across motions
    vmin = np.min(active_vals)
    vmax = np.max(active_vals)

    if vmax > vmin:
        X_norm[active_mask, j] = (active_vals - vmin) / (vmax - vmin)
    else:
        # if all active values are identical, make them all 1.0
        X_norm[active_mask, j] = 1.0
"""
# keep inactive as -1
X_norm[X == -1] = -1

# =========================================================
# GLOBAL FEATURE WEIGHTS
# =========================================================
ANGULAR_WEIGHT = 1.4
PROXIMITY_WEIGHT = 0.8
SPATIAL_WEIGHT = 0.35
HAND_MULTIPLIER = 0.1

def get_feature_weight(feature):
    body, relation, metric = feature.split("|")

    if relation == "angular":
        w = ANGULAR_WEIGHT
    elif relation == "proximity":
        w = PROXIMITY_WEIGHT
    else:
        w = SPATIAL_WEIGHT

    if "hand" in body:
        w *= HAND_MULTIPLIER

    return w

feature_weights = np.array([get_feature_weight(f) for f in ALL_FEATURES], dtype=float)

# =========================================================
# TWO MOTION SCORES
# =========================================================
# 1) normal_score:
#    depends on how many active parameters a motion has
#    (weighted sum)
#
# 2) normalized_score:
#    independent of how many active parameters it has
#    (weighted average over active parameters)

motion_scores_normal = []
#motion_scores_normalized = []

for i, motion_id in enumerate(motion_ids):
    row = X_norm[i]

    active_mask = (row != -1)

    if not np.any(active_mask):
        normal_score = 0.0
        #normalized_score = 0.0
    else:
        vals = row[active_mask]
        weights = feature_weights[active_mask]

        # depends on number of active parameters
        normal_score = np.sum(vals * weights)

        # independent of number of active parameters
        #normalized_score = np.sum(vals * weights) / np.sum(weights)

    motion_scores_normal.append((motion_id, normal_score))
    #motion_scores_normalized.append((motion_id, normalized_score))

# =========================================================
# SORT
# =========================================================
motion_scores_normal.sort(key=lambda x: x[1], reverse=True)
#motion_scores_normalized.sort(key=lambda x: x[1], reverse=True)

# =========================================================
# STATS
# =========================================================
def print_stats(name, motion_scores):
    scores = np.array([s for _, s in motion_scores], dtype=float)
    print(f"\n--- {name} ---")
    print("Mean:", np.mean(scores))
    print("Std:", np.std(scores))
    print("Min:", np.min(scores))
    print("Max:", np.max(scores))

print_stats("NORMAL SCORE (depends on active parameter count)", motion_scores_normal)
#print_stats("NORMALIZED SCORE (independent of active parameter count)", motion_scores_normalized)

# =========================================================
# CDF PLOT
# =========================================================
scores_normal = np.sort(np.array([s for _, s in motion_scores_normal]))
#scores_normalized = np.sort(np.array([s for _, s in motion_scores_normalized]))

n1 = len(scores_normal)
#n2 = len(scores_normalized)

cum1 = np.arange(1, n1 + 1) / n1
#cum2 = np.arange(1, n2 + 1) / n2

plt.plot(scores_normal, cum1, label="normal score")
#plt.plot(scores_normalized, cum2, label="normalized score")
plt.xlabel("Motion Score")
plt.ylabel("Cumulative Relative Frequency")
plt.title("CDF of Motion Scores")
plt.grid()
plt.legend()
plt.show()

# =========================================================
# PRINT SAMPLES
# =========================================================
def print_percentile_samples(title, motion_scores, data):
    n = len(motion_scores)
    points = {
        "100% (top)": 0,
        "75%": int(0.25 * n),
        "50%": int(0.50 * n),
        "25%": int(0.75 * n),
        "0% (bottom)": max(0, n - 5)
    }

    print(f"\n================ {title} ================")
    for label, start in points.items():
        print(f"\n===== {label} =====")
        for k in range(start, min(start + 5, n)):
            m, s = motion_scores[k]
            desc = data[m].get("description", "N/A")
            rank_pct = 100 * (1 - k / (n - 1)) if n > 1 else 100
            print(f"  [{m}] score={s:.4f} rank≈{rank_pct:.2f}% | {desc}")

print_percentile_samples("NORMAL SCORE", motion_scores_normal, data)
#print_percentile_samples("NORMALIZED SCORE", motion_scores_normalized, data)

# =========================================================
# WRITE SCORE BACK TO JSON
# =========================================================
n = len(motion_scores_normal)
for rank, (motion_id, score) in enumerate(motion_scores_normal):
    rank_pct = round(100 * (1 - rank / (n - 1)), 2) if n > 1 else 100.0
    data[motion_id]["normal_score"] = round(score, 6)
    data[motion_id]["rank_pct"] = rank_pct

with open("lower_body_scores.json", "w") as f:
    json.dump(data, f, indent=2)

print(f"\nWrote normal_score to lower_body_scores.json for {len(motion_scores_normal)} motions.")
