#!/usr/bin/env python3
"""
Motioncode 208-D Space Visualization Suite
==========================================
Generates:
  1. PCA per group (with motion ID labels on plot)
  2. t-SNE collective (color by group, legend)
  3. UMAP collective (color by group, legend)
  4. 2D joint-category scatter plots (per group + collective)
  5. 3D joint-category scatter plot (per group + collective)

Output folder structure:
  visualizations/
    pca_per_group/
    tsne/
    umap/
    joint_2d/
    joint_3d/
"""

import os, json, warnings
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

warnings.filterwarnings('ignore')

# ── Load data ────────────────────────────────────────────────────────────────
data = np.load('motioncode_output/vectors.npz', allow_pickle=True)
vectors = data['vectors']        # (5532, 208)
motion_ids = data['motion_ids']  # (5532,)

with open('motioncode_output/metadata.json') as f:
    meta = json.load(f)
labels = meta['labels']

with open('style_labels.json') as f:
    style_labels = json.load(f)

# style_labels is { group: { motion_id: label } } — doubles as group source
verb_groups = {group: list(entries.keys()) for group, entries in style_labels.items()}

# Build motion_id -> row index lookup
id_to_idx = {mid: i for i, mid in enumerate(motion_ids)}

# Build motion_id -> group name lookup (first group wins if duplicates)
id_to_group = {}
for group, ids in verb_groups.items():
    for mid in ids:
        if mid not in id_to_group:
            id_to_group[mid] = group

def get_label(group, mid):
    """Return the style label for a motion, falling back to the motion ID."""
    return style_labels.get(group, {}).get(mid, mid)

# ── Dimension index lookup ───────────────────────────────────────────────────
def find_dim(search_str):
    """Find dimension index by partial label match (case-insensitive)."""
    search_lower = search_str.lower()
    for i, lab in enumerate(labels):
        if search_lower in lab.lower() and '(dup)' not in lab:
            return i
    # fallback: allow dup
    for i, lab in enumerate(labels):
        if search_lower in lab.lower():
            return i
    raise ValueError(f"Dimension not found: {search_str}")

def get_axis_ticks(dim_idx):
    """Return (tick_positions, tick_labels, value_map) for a dimension.
    
    Remaps raw integer values to 0-based category indices so all axes
    have uniform spacing. -1 stays as -1 (inactive), active values
    become 0, 1, 2, ...
    
    Returns:
        positions: list of tick positions [-1, 0, 1, ...]
        tick_labels: list of category name strings
        value_map: dict mapping raw_value -> remapped_position
    """
    lab = labels[dim_idx]
    clean = lab.replace('(dup)', '').strip()
    parts = clean.split('|')
    kind = parts[1]
    val_type = parts[2]

    if val_type == 'velocity':
        cats = meta['categories']['_velocity']['categories']
    else:
        cats = meta['categories'][kind]['intensity_categories']

    # Get actual unique active values in this dimension
    unique_vals = sorted(set(vectors[:, dim_idx].tolist()))
    active_vals = [v for v in unique_vals if v != -1]

    # Build remap: raw value -> 0-based index
    value_map = {-1: -1}
    for i, v in enumerate(active_vals):
        value_map[v] = i

    # Tick positions: -1 then 0..n_active-1
    n = len(active_vals)
    positions = [-1] + list(range(n))

    # Tick labels: match categories to active values
    if n == len(cats):
        tick_labels = ['inactive'] + cats
    elif n < len(cats):
        tick_labels = ['inactive'] + [cats[i] if i < len(cats) else f'{active_vals[i]}'
                                       for i in range(n)]
    else:
        tick_labels = ['inactive'] + [cats[i] if i < len(cats) else f'{active_vals[i]}'
                                       for i in range(n)]

    return positions, tick_labels, value_map

# Pre-compute value maps for all dimensions used in 2D/3D plots
_dim_cache = {}
def get_dim_info(dim_idx):
    """Cached wrapper around get_axis_ticks."""
    if dim_idx not in _dim_cache:
        _dim_cache[dim_idx] = get_axis_ticks(dim_idx)
    return _dim_cache[dim_idx]

def remap_values(raw_vals, dim_idx):
    """Remap raw integer values to 0-based category indices for plotting."""
    _, _, vmap = get_dim_info(dim_idx)
    return np.array([vmap.get(int(v), v) for v in raw_vals], dtype=float)

def apply_ticks_2d(ax, x_dim_idx, y_dim_idx):
    """Set category tick labels on both axes of a 2D plot (using remapped coords)."""
    xpos, xlabels, _ = get_dim_info(x_dim_idx)
    ypos, ylabels, _ = get_dim_info(y_dim_idx)
    ax.set_xticks(xpos)
    ax.set_xticklabels(xlabels, rotation=45, ha='right', fontsize=10)
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.set_xlim(xpos[0] - 0.5, xpos[-1] + 0.5)
    ax.set_ylim(ypos[0] - 0.5, ypos[-1] + 0.5)

def apply_ticks_3d(ax, x_dim_idx, y_dim_idx, z_dim_idx):
    """Set category tick labels and limits on all three axes (using remapped coords)."""
    for dim_idx, setter, label_setter, lim_setter in [
        (x_dim_idx, ax.set_xticks, ax.set_xticklabels, ax.set_xlim),
        (y_dim_idx, ax.set_yticks, ax.set_yticklabels, ax.set_ylim),
        (z_dim_idx, ax.set_zticks, ax.set_zticklabels, ax.set_zlim),
    ]:
        pos, tlabels, _ = get_dim_info(dim_idx)
        lim_setter(pos[0] - 0.5, pos[-1] + 0.5)
        setter(pos)
        label_setter(tlabels, fontsize=8)

# ── Output dirs ──────────────────────────────────────────────────────────────
dirs = [
    'visualizations/pca_per_group',
    'visualizations/tsne',
    'visualizations/umap',
    'visualizations/joint_2d',
    'visualizations/joint_3d',
]
for d in dirs:
    os.makedirs(d, exist_ok=True)

# ── Color palette ────────────────────────────────────────────────────────────
group_names = sorted(verb_groups.keys())
cmap = plt.cm.get_cmap('tab20', max(len(group_names), 20))
# extend with tab20b/tab20c if >20 groups
if len(group_names) > 20:
    cmap2 = plt.cm.get_cmap('tab20b', 20)
    cmap3 = plt.cm.get_cmap('tab20c', 20)
    colors_all = [cmap(i % 20) for i in range(20)] + \
                 [cmap2(i % 20) for i in range(20)] + \
                 [cmap3(i % 20) for i in range(20)]
else:
    colors_all = [cmap(i) for i in range(len(group_names))]
group_color = {g: colors_all[i] for i, g in enumerate(group_names)}

print(f"Loaded {len(vectors)} motions, {len(verb_groups)} groups, {len(labels)} dims")
print(f"Groups: {group_names}")

# ── Helper: replace -1 with 0 for embedding ─────────────────────────────────
vectors_clean = vectors.copy().astype(float)
vectors_clean[vectors_clean == -1] = 0  # treat inactive as zero for embeddings

# ── Build "default" motion set: motions whose style label is exactly "default" ─
default_mids = []  # (motion_id, group_name)
for group, entries in style_labels.items():
    for mid, lbl in entries.items():
        if lbl == 'default' and mid in id_to_idx:
            default_mids.append((mid, group))
print(f"Default-labeled motions: {len(default_mids)}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. PCA PER GROUP (with motion ID labels)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/5] PCA per group...")

for group in group_names:
    ids_in_group = [mid for mid in verb_groups[group] if mid in id_to_idx]
    if len(ids_in_group) < 3:
        print(f"  Skipping {group} (only {len(ids_in_group)} motions)")
        continue

    idxs = [id_to_idx[mid] for mid in ids_in_group]
    X = vectors_clean[idxs]

    pca = PCA(n_components=2, random_state=42)
    X2 = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.scatter(X2[:, 0], X2[:, 1], s=80, alpha=0.6, color=group_color[group])

    # Label every point with style label (or motion ID as fallback)
    for j, mid in enumerate(ids_in_group):
        lbl = get_label(group, mid)
        ax.annotate(lbl, (X2[j, 0], X2[j, 1]), fontsize=12, alpha=0.5,
                    ha='center', va='bottom')

    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}% variance explained)')
    ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}% variance explained)')
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.set_title(f'PCA — {group} ({len(ids_in_group)} motions)')
    fig.tight_layout()
    fig.savefig(f'visualizations/pca_per_group/pca_{group}.png', dpi=200)
    plt.close(fig)
    print(f"  {group}: {len(ids_in_group)} motions, var explained: "
          f"{ev[0]*100:.1f}% + {ev[1]*100:.1f}%")

print("  PCA per group done.")

# --- PCA collective: "default" motions only, labeled by group name ---
if len(default_mids) >= 3:
    d_idxs = [id_to_idx[mid] for mid, _ in default_mids]
    d_groups = [g for _, g in default_mids]
    X_def = vectors_clean[d_idxs]
    pca_def = PCA(n_components=2, random_state=42)
    X_def2 = pca_def.fit_transform(X_def)

    fig, ax = plt.subplots(figsize=(14, 14))
    ax.scatter(X_def2[:, 0], X_def2[:, 1], s=80, alpha=0.6, color='steelblue')
    for j, (mid, g) in enumerate(default_mids):
        ax.annotate(g, (X_def2[j, 0], X_def2[j, 1]), fontsize=12, alpha=0.6,
                    ha='center', va='bottom')
    ev = pca_def.explained_variance_ratio_
    ax.set_xlabel(f'PC1 ({ev[0]*100:.1f}% variance explained)')
    ax.set_ylabel(f'PC2 ({ev[1]*100:.1f}% variance explained)')
    ax.set_xlim(-80, 80)
    ax.set_ylim(-80, 80)
    ax.set_title(f'PCA — Default Motions Only ({len(default_mids)} motions, labeled by group)')
    fig.tight_layout()
    fig.savefig('visualizations/pca_per_group/pca_defaults_collective.png', dpi=200)
    plt.close(fig)
    print("  PCA defaults collective done.")



# ══════════════════════════════════════════════════════════════════════════════
# 2. t-SNE COLLECTIVE (color by group, legend)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/5] t-SNE collective...")

# Only include motions that belong to a group
grouped_mask = np.array([mid in id_to_group for mid in motion_ids])
grouped_idxs = np.where(grouped_mask)[0]
X_grouped = vectors_clean[grouped_idxs]
mids_grouped = motion_ids[grouped_idxs]
groups_grouped = [id_to_group[mid] for mid in mids_grouped]

tsne = TSNE(n_components=2, random_state=42,
            perplexity=min(max(5, int(np.sqrt(len(X_grouped)))), len(X_grouped) - 1),
            max_iter=1000, init='pca', learning_rate='auto')
X_tsne = tsne.fit_transform(X_grouped)

fig, ax = plt.subplots(figsize=(16, 12))
for g in group_names:
    mask = np.array([gr == g for gr in groups_grouped])
    if mask.sum() == 0:
        continue
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], s=32, alpha=0.5,
              color=group_color[g], label=g)

ax.set_title(f't-SNE of All Grouped Motions ({len(X_grouped)} motions, {len(group_names)} groups)')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7,
          markerscale=2, ncol=2, framealpha=0.9)
fig.tight_layout()
fig.savefig('visualizations/tsne/tsne_collective.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("  t-SNE done.")


# ══════════════════════════════════════════════════════════════════════════════
# 3. UMAP COLLECTIVE (color by group, legend)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/5] UMAP collective...")

reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
X_umap = reducer.fit_transform(X_grouped)

fig, ax = plt.subplots(figsize=(16, 12))
for g in group_names:
    mask = np.array([gr == g for gr in groups_grouped])
    if mask.sum() == 0:
        continue
    ax.scatter(X_umap[mask, 0], X_umap[mask, 1], s=32, alpha=0.5,
              color=group_color[g], label=g)

ax.set_title(f'UMAP of All Grouped Motions ({len(X_grouped)} motions, {len(group_names)} groups)')
ax.set_xlabel('UMAP 1')
ax.set_ylabel('UMAP 2')
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7,
          markerscale=2, ncol=2, framealpha=0.9)
fig.tight_layout()
fig.savefig('visualizations/umap/umap_collective.png', dpi=200, bbox_inches='tight')
plt.close(fig)
print("  UMAP done.")


# ══════════════════════════════════════════════════════════════════════════════
# 4. 2D JOINT-CATEGORY SCATTER PLOTS
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] 2D joint-category plots...")

# Define the 5 pairs
pairs_2d = [
    {
        'x_search': 'left_hip+left_knee+left_ankle|angular|intensity',
        'y_search': 'translation|displacement_z|intensity',
        'title': 'Left Knee Angular Intensity vs Displacement Z Intensity',
        'fname': 'lknee_angular_vs_disp_z',
    },
    {
        'x_search': 'left_hip+left_knee+left_ankle|angular|intensity',
        'y_search': 'translation|displacement_y|intensity',
        'title': 'Left Knee Angular Intensity vs Displacement Y Intensity',
        'fname': 'lknee_angular_vs_disp_y',
    },
    {
        'x_search': 'left_hip+left_knee+left_ankle|angular|velocity',
        'y_search': 'right_hip+right_knee+right_ankle|angular|velocity',
        'title': 'Left Knee Angular Velocity vs Right Knee Angular Velocity',
        'fname': 'lknee_vel_vs_rknee_vel',
    },
    {
        'x_search': 'left_foot+right_foot|proximity|intensity',
        'y_search': 'translation|displacement_z|intensity',
        'title': '(Left Foot, Right Foot) Proximity Intensity vs Displacement Z Intensity',
        'fname': 'foot_prox_vs_disp_z',
    },
    {
        'x_search': 'orientation|rotation_yaw|intensity',
        'y_search': 'translation|displacement_x|intensity',
        'title': 'Rotation Yaw Intensity vs Displacement X Intensity',
        'fname': 'rot_yaw_vs_disp_x',
    },
]

# Resolve dimension indices
for p in pairs_2d:
    p['x_idx'] = find_dim(p['x_search'])
    p['y_idx'] = find_dim(p['y_search'])
    p['x_label'] = labels[p['x_idx']]
    p['y_label'] = labels[p['y_idx']]
    print(f"  {p['fname']}: dim {p['x_idx']} vs dim {p['y_idx']}")

# --- Per-group 2D plots (with motion ID labels) ---
for pair in pairs_2d:
    pair_dir = f"visualizations/joint_2d/{pair['fname']}"
    os.makedirs(pair_dir, exist_ok=True)

    for group in group_names:
        ids_in_group = [mid for mid in verb_groups[group] if mid in id_to_idx]
        if len(ids_in_group) < 2:
            continue
        idxs = [id_to_idx[mid] for mid in ids_in_group]
        xvals = remap_values(vectors[idxs, pair['x_idx']], pair['x_idx'])
        yvals = remap_values(vectors[idxs, pair['y_idx']], pair['y_idx'])

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(xvals, yvals, s=80, alpha=0.6, color=group_color[group])
        for j, mid in enumerate(ids_in_group):
            lbl = get_label(group, mid)
            ax.annotate(lbl, (xvals[j], yvals[j]), fontsize=12, alpha=0.5,
                        ha='center', va='bottom')
        ax.set_xlabel(pair['x_label'])
        ax.set_ylabel(pair['y_label'])
        ax.set_title(f"{pair['title']}\nGroup: {group} ({len(ids_in_group)} motions)")
        apply_ticks_2d(ax, pair['x_idx'], pair['y_idx'])
        fig.tight_layout()
        fig.savefig(f"{pair_dir}/{group}.png", dpi=180)
        plt.close(fig)

    # --- Collective 2D plot (color by group, no labels) ---
    fig, ax = plt.subplots(figsize=(14, 10))
    for g in group_names:
        ids_g = [mid for mid in verb_groups[g] if mid in id_to_idx]
        if not ids_g:
            continue
        idxs_g = [id_to_idx[mid] for mid in ids_g]
        xv = remap_values(vectors[idxs_g, pair['x_idx']], pair['x_idx'])
        yv = remap_values(vectors[idxs_g, pair['y_idx']], pair['y_idx'])
        ax.scatter(xv, yv, s=40, alpha=0.4, color=group_color[g], label=g)

    ax.set_xlabel(pair['x_label'])
    ax.set_ylabel(pair['y_label'])
    ax.set_title(f"{pair['title']} — All Groups")
    apply_ticks_2d(ax, pair['x_idx'], pair['y_idx'])
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6,
              markerscale=2, ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{pair_dir}/collective.png", dpi=180, bbox_inches='tight')
    plt.close(fig)

    # --- Collective 2D: "default" motions only, labeled by group name ---
    d_ids_valid = [(mid, g) for mid, g in default_mids if mid in id_to_idx]
    if len(d_ids_valid) >= 2:
        d_idxs = [id_to_idx[mid] for mid, _ in d_ids_valid]
        xv_def = remap_values(vectors[d_idxs, pair['x_idx']], pair['x_idx'])
        yv_def = remap_values(vectors[d_idxs, pair['y_idx']], pair['y_idx'])
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.scatter(xv_def, yv_def, s=80, alpha=0.6, color='steelblue')
        for j, (mid, g) in enumerate(d_ids_valid):
            ax.annotate(g, (xv_def[j], yv_def[j]), fontsize=12, alpha=0.6,
                        ha='center', va='bottom')
        ax.set_xlabel(pair['x_label'])
        ax.set_ylabel(pair['y_label'])
        ax.set_title(f"{pair['title']} — Default Motions Only (labeled by group)")
        apply_ticks_2d(ax, pair['x_idx'], pair['y_idx'])
        fig.tight_layout()
        fig.savefig(f"{pair_dir}/defaults_collective.png", dpi=180)
        plt.close(fig)

print("  2D joint plots done.")


# ══════════════════════════════════════════════════════════════════════════════
# 5. 3D JOINT-CATEGORY SCATTER PLOT
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/5] 3D joint-category plot...")

triple_3d = {
    'x_search': 'left_hip+left_knee+left_ankle|angular|intensity',
    'y_search': 'translation|displacement_z|intensity',
    'z_search': 'translation|displacement_y|intensity',
    'title': 'Left Knee Angular × Displacement Z × Displacement Y (Intensity)',
    'fname': 'lknee_angular_x_disp_z_x_disp_y',
}
triple_3d['x_idx'] = find_dim(triple_3d['x_search'])
triple_3d['y_idx'] = find_dim(triple_3d['y_search'])
triple_3d['z_idx'] = find_dim(triple_3d['z_search'])
triple_3d['x_label'] = labels[triple_3d['x_idx']]
triple_3d['y_label'] = labels[triple_3d['y_idx']]
triple_3d['z_label'] = labels[triple_3d['z_idx']]

trip_dir = f"visualizations/joint_3d/{triple_3d['fname']}"
os.makedirs(trip_dir, exist_ok=True)

# --- Per-group 3D (with labels) ---
for group in group_names:
    ids_in_group = [mid for mid in verb_groups[group] if mid in id_to_idx]
    if len(ids_in_group) < 2:
        continue
    idxs = [id_to_idx[mid] for mid in ids_in_group]
    xv = remap_values(vectors[idxs, triple_3d['x_idx']], triple_3d['x_idx'])
    yv = remap_values(vectors[idxs, triple_3d['y_idx']], triple_3d['y_idx'])
    zv = remap_values(vectors[idxs, triple_3d['z_idx']], triple_3d['z_idx'])

    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xv, yv, zv, s=80, alpha=0.6, color=group_color[group])
    for j, mid in enumerate(ids_in_group):
        lbl = get_label(group, mid)
        ax.text(xv[j], yv[j], zv[j], lbl, fontsize=10, alpha=0.4)
    ax.set_xlabel(triple_3d['x_label'], fontsize=7)
    ax.set_ylabel(triple_3d['y_label'], fontsize=7)
    ax.set_zlabel(triple_3d['z_label'], fontsize=7)
    ax.set_title(f"{triple_3d['title']}\nGroup: {group} ({len(ids_in_group)} motions)",
                 fontsize=10)
    apply_ticks_3d(ax, triple_3d['x_idx'], triple_3d['y_idx'], triple_3d['z_idx'])
    fig.tight_layout()
    fig.savefig(f"{trip_dir}/{group}.png", dpi=180)
    plt.close(fig)

# --- Collective 3D (color by group, no labels) ---
fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')
for g in group_names:
    ids_g = [mid for mid in verb_groups[g] if mid in id_to_idx]
    if not ids_g:
        continue
    idxs_g = [id_to_idx[mid] for mid in ids_g]
    xv = remap_values(vectors[idxs_g, triple_3d['x_idx']], triple_3d['x_idx'])
    yv = remap_values(vectors[idxs_g, triple_3d['y_idx']], triple_3d['y_idx'])
    zv = remap_values(vectors[idxs_g, triple_3d['z_idx']], triple_3d['z_idx'])
    ax.scatter(xv, yv, zv, s=32, alpha=0.4, color=group_color[g], label=g)

ax.set_xlabel(triple_3d['x_label'], fontsize=7)
ax.set_ylabel(triple_3d['y_label'], fontsize=7)
ax.set_zlabel(triple_3d['z_label'], fontsize=7)
ax.set_title(f"{triple_3d['title']} — All Groups", fontsize=10)
apply_ticks_3d(ax, triple_3d['x_idx'], triple_3d['y_idx'], triple_3d['z_idx'])
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=5,
          markerscale=2, ncol=2, framealpha=0.9)
fig.tight_layout()
fig.savefig(f"{trip_dir}/collective.png", dpi=180, bbox_inches='tight')
plt.close(fig)

# --- Collective 3D: "default" motions only, labeled by group name ---
d_ids_valid = [(mid, g) for mid, g in default_mids if mid in id_to_idx]
if len(d_ids_valid) >= 2:
    d_idxs = [id_to_idx[mid] for mid, _ in d_ids_valid]
    xv_def = remap_values(vectors[d_idxs, triple_3d['x_idx']], triple_3d['x_idx'])
    yv_def = remap_values(vectors[d_idxs, triple_3d['y_idx']], triple_3d['y_idx'])
    zv_def = remap_values(vectors[d_idxs, triple_3d['z_idx']], triple_3d['z_idx'])
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xv_def, yv_def, zv_def, s=80, alpha=0.6, color='steelblue')
    for j, (mid, g) in enumerate(d_ids_valid):
        ax.text(xv_def[j], yv_def[j], zv_def[j], g, fontsize=10, alpha=0.5)
    ax.set_xlabel(triple_3d['x_label'], fontsize=7)
    ax.set_ylabel(triple_3d['y_label'], fontsize=7)
    ax.set_zlabel(triple_3d['z_label'], fontsize=7)
    ax.set_title(f"{triple_3d['title']} — Default Motions Only (labeled by group)", fontsize=10)
    apply_ticks_3d(ax, triple_3d['x_idx'], triple_3d['y_idx'], triple_3d['z_idx'])
    fig.tight_layout()
    fig.savefig(f"{trip_dir}/defaults_collective.png", dpi=180)
    plt.close(fig)

print("  3D joint plots done.")

# ══════════════════════════════════════════════════════════════════════════════
print("\n✓ All visualizations saved to visualizations/")
