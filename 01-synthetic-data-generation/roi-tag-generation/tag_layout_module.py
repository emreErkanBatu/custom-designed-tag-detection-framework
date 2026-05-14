# -*- coding: utf-8 -*-

# tag_layout_module.py
# -*- coding: utf-8 -*-
"""
Tag layout module.

Task:
- Given a scene size and a list of tag images, generate random
  non-overlapping tag placements.
"""

import random
import os

# Range for the number of tags to place
MIN_TAGS = 20
MAX_TAGS = 30

# Camera parameters, kept fixed in this script
DEFAULT_FOV_DEG = 60.0
DEFAULT_DIST_SCALE = 2.0

# Collision-check parameters
COLLISION_PADDING = 2               # Extra spacing between boxes
EFFECTIVE_SIZE_FACTOR = 0.85        # Slightly shrink the box during collision checking


def rects_intersect(r1, r2, padding=0):
    """
    Check whether two axis-aligned rectangles intersect.
    r = (xmin, ymin, xmax, ymax)
    """
    x1_min, y1_min, x1_max, y1_max = r1
    x2_min, y2_min, x2_max, y2_max = r2

    x1_min -= padding
    y1_min -= padding
    x1_max += padding
    y1_max += padding

    x2_min -= padding
    y2_min -= padding
    x2_max += padding
    y2_max += padding

    if x1_max <= x2_min or x2_max <= x1_min:
        return False
    if y1_max <= y2_min or y2_max <= y1_min:
        return False
    return True


def _resolve_rng(rng=None, rng_seed=None):
    if rng is not None:
        return rng
    if rng_seed is not None:
        return random.Random(rng_seed)
    return random


def _randint(rng_obj, low, high_inclusive):
    if hasattr(rng_obj, "integers"):
        return int(rng_obj.integers(low, high_inclusive + 1))
    return int(rng_obj.randint(low, high_inclusive))


def _choice(rng_obj, items):
    if hasattr(rng_obj, "choice"):
        return rng_obj.choice(items)
    return rng_obj.choice(items)


def generate_random_layout(scene_shape,
                           tag_paths,
                           min_size,
                           max_size,
                           min_tags=MIN_TAGS,
                           max_tags=MAX_TAGS,
                           rng_seed=None,
                           rng=None):
    """
    Generate random, non-overlapping tag placements.

    Parameters:
        scene_shape : (H, W, C)
        tag_paths   : list of tag image paths
        min_size/max_size : tag-size limits in pixels
        min_tags/max_tags : target range for the number of tags
        rng_seed    : optional seed
        rng         : optional shared RNG; numpy Generator is supported

    Returns:
        configs : list[dict]
    """
    rng_obj = _resolve_rng(rng=rng, rng_seed=rng_seed)

    h, w, _ = scene_shape

    if not tag_paths:
        return []

    if min_size <= 0 or max_size <= 0 or min_size > max_size:
        raise ValueError(f"Invalid min_size/max_size: {min_size}, {max_size}")

    target_tags = _randint(rng_obj, min_tags, max_tags)

    configs = []
    placed_rects = []

    if target_tags <= 1:
        base_sizes = [max_size]
    else:
        step = (max_size - min_size) / max(target_tags - 1, 1)
        base_sizes = [int(max_size - i * step) for i in range(target_tags)]

    size_candidates = [
        max(min_size, min(max_size, s + _randint(rng_obj, -15, 15)))
        for s in base_sizes
    ]

    for nominal_size in size_candidates:
        size = int(max(min_size, min(max_size, nominal_size)))

        margin = 5
        min_cx = margin + size // 2
        max_cx = w - margin - size // 2
        min_cy = margin + size // 2
        max_cy = h - margin - size // 2

        if min_cx >= max_cx or min_cy >= max_cy:
            continue

        placed = False
        size_levels = [size, int(size * 0.9), int(size * 0.8)]

        for size_level in size_levels:
            size_level = max(min_size, min(max_size, size_level))
            half = size_level // 2
            if half <= 0:
                continue

            min_cx = margin + half
            max_cx = w - margin - half
            min_cy = margin + half
            max_cy = h - margin - half

            if min_cx >= max_cx or min_cy >= max_cy:
                continue

            for _try in range(300):
                cx = _randint(rng_obj, min_cx, max_cx)
                cy = _randint(rng_obj, min_cy, max_cy)

                eff_half = int(half * EFFECTIVE_SIZE_FACTOR)
                rect = (cx - eff_half, cy - eff_half, cx + eff_half, cy + eff_half)

                collision = any(
                    rects_intersect(rect, r, padding=COLLISION_PADDING)
                    for r in placed_rects
                )
                if collision:
                    continue

                placed_rects.append(rect)

                yaw_deg = _randint(rng_obj, -50, 50)
                pitch_deg = _randint(rng_obj, -50, 50)
                roll_deg = _randint(rng_obj, 0, 359)

                tag_path = _choice(rng_obj, tag_paths)
                name = os.path.splitext(os.path.basename(tag_path))[0]

                cfg = {
                    "name": name,
                    "tag_path": tag_path,
                    "target_size_px": size_level,
                    "center_xy": (cx, cy),
                    "yaw_deg": yaw_deg,
                    "pitch_deg": pitch_deg,
                    "roll_deg": roll_deg,
                    "fov_deg": DEFAULT_FOV_DEG,
                    "dist_scale": DEFAULT_DIST_SCALE,
                }
                configs.append(cfg)
                placed = True
                break

            if placed:
                break

    return configs
