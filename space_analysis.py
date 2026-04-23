"""
Data analysis of the motioncode dimensional space.
"""
import json, os
import numpy as np

# Load motioncode space
data = np.load('motioncode_output/vectors.npz', allow_pickle=True)
vectors = data['vectors'].astype(np.float32)
motion_ids = data['motion_ids'].tolist()
id2idx = {mid: i for i, mid in enumerate(motion_ids)}

# Load verb groups
with open('verb_groups copy.json') as f:
    verb_groups = json.load(f)

D = vectors.shape[1]  # 208

################################################################################
# HELPER FUNCTIONS
################################################################################

def compute_stats(vecs):
    """Compute stats for a set of vectors."""
    mean = vecs.mean(axis=0)
    std = vecs.std(axis=0)
    mins = vecs.min(axis=0)
    maxs = vecs.max(axis=0)
    ranges = maxs - mins

    # Continuous volume: product of ranges per dimension (only non-zero ranges)
    nonzero_ranges = ranges[ranges > 0]
    if len(nonzero_ranges) > 0:
        log_volume = np.sum(np.log(nonzero_ranges))  # log to avoid overflow
        volume_log = log_volume
    else:
        volume_log = 0.0

    # Discrete count: number of unique coordinate vectors
    unique_points = len(set(map(tuple, vecs.astype(int))))

    # Number of active dimensions (at least one non -1 value across all motions)
    active_dims = np.sum(np.any(vecs != -1, axis=0))

    # Per-dimension unique value counts
    unique_per_dim = np.array([len(np.unique(vecs[:, d])) for d in range(vecs.shape[1])])

    # Discrete volume: product of unique values per active dimension
    active_unique = unique_per_dim[unique_per_dim > 1]
    if len(active_unique) > 0:
        log_discrete_volume = np.sum(np.log(active_unique.astype(float)))
    else:
        log_discrete_volume = 0.0

    return {
        "num_motions": len(vecs),
        "mean_per_dim": mean,
        "std_per_dim": std,
        "min_per_dim": mins,
        "max_per_dim": maxs,
        "range_per_dim": ranges,
        "active_dimensions": int(active_dims),
        "total_dimensions": int(vecs.shape[1]),
        "unique_points_discrete": unique_points,
        "unique_per_dim": unique_per_dim,
        "log_continuous_volume": float(volume_log),
        "log_discrete_volume": float(log_discrete_volume),
    }


def summarize(stats, label):
    """Print a summary of stats."""
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"{'='*70}")
    print(f"  Motions:              {stats['num_motions']}")
    print(f"  Active dimensions:    {stats['active_dimensions']} / {stats['total_dimensions']}")
    print(f"  Unique points:        {stats['unique_points_discrete']}")
    print(f"  Log continuous vol:   {stats['log_continuous_volume']:.2f}")
    print(f"  Log discrete vol:     {stats['log_discrete_volume']:.2f}")
    print(f"  Mean coord (avg):     {stats['mean_per_dim'][stats['mean_per_dim'] != -1].mean():.2f}")
    print(f"  Std coord (avg):      {stats['std_per_dim'][stats['std_per_dim'] > 0].mean():.2f}")
    print(f"  Avg range per dim:    {stats['range_per_dim'].mean():.2f}")


################################################################################
# GLOBAL ANALYSIS (all motions)
################################################################################
print("\n" + "#"*70)
print("# GLOBAL SPACE ANALYSIS")
print("#"*70)

global_stats = compute_stats(vectors)
summarize(global_stats, "ALL MOTIONS")

################################################################################
# PER-GROUP ANALYSIS
################################################################################
print("\n" + "#"*70)
print("# PER-GROUP ANALYSIS")
print("#"*70)

group_results = {}
for group_name, file_ids in sorted(verb_groups.items()):
    # Handle mixed formats: some entries may be dicts with 'file' key
    clean_ids = []
    for fid in file_ids:
        if isinstance(fid, dict):
            clean_ids.append(fid.get('file', ''))
        else:
            clean_ids.append(fid)
    indices = [id2idx[fid] for fid in clean_ids if fid in id2idx]
    if len(indices) < 2:
        continue
    gvecs = vectors[indices]
    stats = compute_stats(gvecs)
    summarize(stats, f"GROUP: {group_name}")
    group_results[group_name] = stats

################################################################################
# CUMULATIVE RELATIVE FREQUENCY: space coverage
################################################################################
print("\n" + "#"*70)
print("# SPACE COVERAGE: CUMULATIVE RELATIVE FREQUENCY")
print("#"*70)

# Method 1: Discrete — unique points as fraction of total unique points
global_unique = global_stats['unique_points_discrete']
print(f"\n{'Group':<20} {'Motions':>8} {'Unique Pts':>12} {'% of Global':>12} "
      f"{'Log Cont Vol':>14} {'% Log Vol':>10}")
print("-" * 80)

for group_name in sorted(group_results.keys(), key=lambda k: group_results[k]['num_motions'], reverse=True):
    gs = group_results[group_name]
    pct_discrete = (gs['unique_points_discrete'] / global_unique) * 100
    pct_log_vol = (gs['log_continuous_volume'] / global_stats['log_continuous_volume']) * 100 if global_stats['log_continuous_volume'] > 0 else 0
    print(f"{group_name:<20} {gs['num_motions']:>8} {gs['unique_points_discrete']:>12} "
          f"{pct_discrete:>11.2f}% {gs['log_continuous_volume']:>14.2f} {pct_log_vol:>9.2f}%")

# Totals
total_group_unique = sum(gs['unique_points_discrete'] for gs in group_results.values())
print("-" * 80)
print(f"{'TOTAL (groups)':<20} {sum(gs['num_motions'] for gs in group_results.values()):>8} "
      f"{total_group_unique:>12} {(total_group_unique/global_unique)*100:>11.2f}%")
print(f"{'GLOBAL':<20} {global_stats['num_motions']:>8} {global_unique:>12} {'100.00%':>12}")

################################################################################
# SAVE DETAILED RESULTS
################################################################################
output = {
    "global": {
        "num_motions": global_stats['num_motions'],
        "active_dimensions": global_stats['active_dimensions'],
        "total_dimensions": global_stats['total_dimensions'],
        "unique_points": global_stats['unique_points_discrete'],
        "log_continuous_volume": global_stats['log_continuous_volume'],
        "log_discrete_volume": global_stats['log_discrete_volume'],
        "mean_coord_avg": float(global_stats['mean_per_dim'][global_stats['mean_per_dim'] != -1].mean()),
        "std_coord_avg": float(global_stats['std_per_dim'][global_stats['std_per_dim'] > 0].mean()),
    },
    "groups": {}
}
for gn, gs in group_results.items():
    output["groups"][gn] = {
        "num_motions": gs['num_motions'],
        "active_dimensions": gs['active_dimensions'],
        "unique_points": gs['unique_points_discrete'],
        "log_continuous_volume": gs['log_continuous_volume'],
        "log_discrete_volume": gs['log_discrete_volume'],
        "pct_of_global_discrete": (gs['unique_points_discrete'] / global_unique) * 100,
        "pct_of_global_log_vol": (gs['log_continuous_volume'] / global_stats['log_continuous_volume']) * 100 if global_stats['log_continuous_volume'] > 0 else 0,
        "mean_coord_avg": float(gs['mean_per_dim'][gs['mean_per_dim'] != -1].mean()),
        "std_coord_avg": float(gs['std_per_dim'][gs['std_per_dim'] > 0].mean()),
    }

with open('space_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nDetailed results saved to space_analysis.json")
