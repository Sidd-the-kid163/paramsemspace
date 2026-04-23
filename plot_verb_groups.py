"""
For each verb group: find vectors, compute centroid, find nearest file(s) to centroid.
"""
import json, os
import numpy as np

# Load motioncode space
data = np.load('motioncode_output/vectors.npz', allow_pickle=True)
vectors = data['vectors']
motion_ids = data['motion_ids'].tolist()
id2idx = {mid: i for i, mid in enumerate(motion_ids)}

# Load verb groups
with open('verb_groups.json') as f:
    verb_groups = json.load(f)

results = {}

for group_name, file_ids in verb_groups.items():
    indices = [id2idx[fid] for fid in file_ids if fid in id2idx]
    if len(indices) < 2:
        print(f"  {group_name}: skipped ({len(indices)} files found)")
        continue

    group_vectors = vectors[indices].astype(np.float32)
    group_ids = [fid for fid in file_ids if fid in id2idx]

    centroid = group_vectors.mean(axis=0)
    dists = np.linalg.norm(group_vectors - centroid, axis=1)

    top3_idx = np.argsort(dists)[:3]
    top3 = [(group_ids[i], float(dists[i])) for i in top3_idx]

    results[group_name] = {
        "num_files": len(indices),
        "center_file": top3[0][0],
        "center_distance": top3[0][1],
        "top3_nearest": top3,
        "mean_distance": float(dists.mean()),
        "std_distance": float(dists.std()),
    }

    print(f"  {group_name}: {len(indices)} files, center={top3[0][0]} (dist={top3[0][1]:.2f})")

with open('verb_group_centers.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nSaved {len(results)} groups to verb_group_centers.json")
