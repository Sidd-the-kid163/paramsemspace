```
                                    D = Σ(solo kinds)[joint_sets × 2] + Σ(pairwise kinds)[joint_sets × 2 × 2] = 200
                                    ─────────────────────────────────────────────────────────────────────────────────
                                         │                          │                              │
                                         │                          │                              │
                                    ─────┴─────                ─────┴─────                    ─────┴─────
                                    │  × 2     │                │  × 2     │                    │  × 2     │
                                    │(per slot)│                │(per slot)│                    │(extra)   │
                                    ───┬───┬───                ───┬───┬───                    ───────────
                                       │   │                      │   │                            │
                              ┌────────┘   └────────┐    ┌───────┘   └───────┐                    │
                              ▼                      ▼    ▼                    ▼                    ▼
                         INTENSITY              VELOCITY  INTENSITY       VELOCITY         Holds pair of joints
                      (varies per kind)       (shared)   (varies per kind) (shared)            (relative)
                                                  │
                                                  │
                                                  ▼
                                          ┌───────────────┐
                                          │  5 categories  │
                                          │  very_slow     │
                                          │  slow          │
                                          │  moderate      │
                                          │  fast          │
                                          │  very_fast     │
                                          └───────────────┘


    ════════════════════════════════════════════════════════════════════════════════════════════════════════
                    SOLO KINDS                          │              PAIRWISE KINDS
         Σ [joint_sets × 2] = 20                       │         Σ [joint_sets × 2 × 2] = 180
    ════════════════════════════════════════════════════════════════════════════════════════════════════════
                                                       │
     angular              (4 joints)                   │    proximity              (22 joint pairs)
       7 intensity levels                              │      5 intensity levels
       significant_bend, moderate_bend,                │      significant_closing, moderate_closing,
       slight_bend, no_action,                         │      stationary, moderate_spreading,
       slight_extension, moderate_extension,            │      significant_spreading
       significant_extension                           │
                                                       │    spatial_relation_x     (5 joint pairs)
     displacement_x       (1 joint)                    │      3 intensity levels
       11 intensity levels                             │      left-to-right, stationary,
       very_long_left, long_left, moderate_left,       │      right-to-left
       short_left, very_short_left, no_action,         │
       very_short_right, short_right,                  │    spatial_relation_y     (13 joint pairs)
       moderate_right, long_right, very_long_right     │      3 intensity levels
                                                       │      above-to-below, stationary,
     displacement_y       (1 joint)                    │      below-to-above
       11 intensity levels                             │
       very_long_down ... no_action ... very_long_up   │    spatial_relation_z     (5 joint pairs)
                                                       │      3 intensity levels
     displacement_z       (1 joint)                    │      front-to-behind, stationary,
       11 intensity levels                             │      behind-to-front
       very_long_backward ... no_action ...            │
       very_long_forward                               │
                                                       │
     rotation_pitch       (1 joint)                    │
       7 intensity levels                              │
       significant_leaning_backward ...                │
       no_action ... significant_leaning_forward       │
                                                       │
     rotation_roll        (1 joint)                    │
       7 intensity levels                              │
       significant_leaning_right ...                   │
       no_action ... significant_leaning_left          │
                                                       │
     rotation_yaw         (1 joint)                    │
       7 intensity levels                              │
       significant_turn_clockwise ...                  │
       no_action ... significant_turn_counterclockwise │
    ════════════════════════════════════════════════════════════════════════════════════════════════════════
```
