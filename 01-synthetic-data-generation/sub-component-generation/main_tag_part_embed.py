# main_tag_embed_with_xml_UPDATED_v3_perBG_composite.py
# -*- coding: utf-8 -*-
"""
Synthetic data generation for sub-component detection.

- The individual sub-component generation stage fills per-background targets for S1-S4.
- The composite generation stage places a full tag made of four pieces as a single visual unit.
  In the XML annotation, the four pieces inside the full tag are saved as separate objects.

Note:
- TAG_DIR must contain four tag-piece images. If A/B/C/D names are available, they are used directly; otherwise, the first four files in alphabetical order are used.
  The 2x2 layout order is: [0]=top-left, [1]=top-right, [2]=bottom-left, [3]=bottom-right.
"""

import os
import cv2
import csv
import xml.etree.ElementTree as ET
import numpy as np

from tag_layout_module import generate_random_layout
from anti_overfitting_tag_embed import (
    embed_tag_anti_overfitting,
    anti_overfitting_overlay,
    compute_tight_bbox_from_mask,
    add_bbox_augmentation,
)
from tag_embed_module import warp_and_crop_with_mask  # Kept only as a reference for the earlier flow


# -------------------------
# IO
# -------------------------
SCENE_DIR = "scenes"
TAG_DIR   = "tags"

OUTPUT_DIR = r"E:\TFOD\datasets\04_tag_part\images"
TRAIN_DIR = os.path.join(OUTPUT_DIR, "train")
TEST_DIR  = os.path.join(OUTPUT_DIR, "test")

ENABLE_JPEG_COMPAT = True
JPEG_QUALITY_RANGE = (72, 96)

# DEBUG
DEBUG_DRAW = False
DEBUG_THICKNESS = 2
DEBUG_FONT_SCALE = 0.55

ALPHA_FORCE_METHOD = "gauss"  # "gauss" / "dist" / "morph" / "random"
PIPELINE_MODE = "full"  # "full" / "naive_same_scale" / "no_bg_aware_block" / "no_appearance_realism_block"

# Background patch size for each generated sample
CROP_SIZE = 512


# -------------------------
# Random crop
# -------------------------
def random_crop(scene_bgr: np.ndarray, crop_size: int, rng: np.random.Generator) -> np.ndarray:
    """Return a random crop (crop_size x crop_size) from the given background image."""
    h, w = scene_bgr.shape[:2]
    if h < crop_size or w < crop_size:
        raise ValueError(f"Scene too small for {crop_size}x{crop_size} crop: {w}x{h}")
    x0 = int(rng.integers(0, w - crop_size + 1))
    y0 = int(rng.integers(0, h - crop_size + 1))
    return scene_bgr[y0:y0 + crop_size, x0:x0 + crop_size].copy()


def ensure_dirs():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)


# -------------------------
# Size ranges
# -------------------------
# Size ranges for the individual sub-component generation stage
SIZE_RANGES = [
    (80,  200),
    (200, 300),
    (300, 350),
    (350, 400),
]

# Size ranges for the full-tag composite generation stage
COMPOSITE_SIZE_RANGES = [
    (150, 200),  # S1
    (200, 300),  # S2
    (300, 350),  # S3
    (350, 400),  # S4
]

MAX_TRIES_PER_RANGE = 200   

# Target full-tag composite counts
COMPOSITE_TARGET_MAIN_TAGS = {1: 40, 2: 20, 3: 20, 4: 20}  # Total of 100 full-tag composites

COMPOSITE_MIN_TAGS_PER_IMAGE = 1
COMPOSITE_MAX_TAGS_PER_IMAGE = 6  # Multiple full-tag composites can be placed in the same image when conditions allow

# Piece size and spacing used to build the full-tag composite
PIECE_SIZE_PX = 500
PIECE_GAP_PX  = 36


# -------------------------
# Global counters & reports
# -------------------------
train_tag_count = {}
test_tag_count  = {}
train_image_count = 0
test_image_count  = 0



# Separate counters for the full-tag composite stage
composite_train_image_count = 0
composite_test_image_count  = 0
composite_train_tag_count = {}
composite_test_tag_count  = {}
def list_images_in_dir(directory, exts=(".png", ".jpg", ".jpeg", ".bmp")):
    files = []
    if os.path.exists(directory):
        for fname in os.listdir(directory):
            if fname.lower().endswith(exts):
                files.append(os.path.join(directory, fname))
    return sorted(files)


def write_voc_xml(image_path, image_shape, objects, xml_path):
    h, w, c = image_shape

    annotation = ET.Element("annotation")
    ET.SubElement(annotation, "folder").text = os.path.basename(os.path.dirname(image_path))
    ET.SubElement(annotation, "filename").text = os.path.basename(image_path)
    ET.SubElement(annotation, "path").text = os.path.abspath(image_path)

    source = ET.SubElement(annotation, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(annotation, "size")
    ET.SubElement(size, "width").text = str(w)
    ET.SubElement(size, "height").text = str(h)
    ET.SubElement(size, "depth").text = str(c)

    ET.SubElement(annotation, "segmented").text = "0"

    for obj in objects:
        name = obj["name"]
        xmin, ymin, xmax, ymax = obj["bbox"]

        obj_el = ET.SubElement(annotation, "object")
        ET.SubElement(obj_el, "name").text = name

        bnd = ET.SubElement(obj_el, "bndbox")
        ET.SubElement(bnd, "xmin").text = str(int(xmin))
        ET.SubElement(bnd, "ymin").text = str(int(ymin))
        ET.SubElement(bnd, "xmax").text = str(int(xmax))
        ET.SubElement(bnd, "ymax").text = str(int(ymax))

    os.makedirs(os.path.dirname(xml_path), exist_ok=True)
    ET.ElementTree(annotation).write(xml_path, encoding="utf-8", xml_declaration=True)


def update_tag_counts(objects, is_train: bool):
    target_dict = train_tag_count if is_train else test_tag_count
    for obj in objects:
        name = obj["name"]
        target_dict[name] = target_dict.get(name, 0) + 1


def save_report_csv(csv_path, image_count, tag_count_dict):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Total Images", image_count])
        writer.writerow([])
        writer.writerow(["Tag Name", "Total BBox Count"])
        for tag, c in sorted(tag_count_dict.items()):
            writer.writerow([tag, c])
    print(f"CSV saved -> {csv_path}")


def jpeg_encode_decode(bgr: np.ndarray, quality: int) -> np.ndarray:
    quality = int(np.clip(quality, 30, 100))
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return bgr
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else bgr


def draw_debug_box(img, bbox, label: str):
    x0, y0, x1, y1 = [int(v) for v in bbox]
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), DEBUG_THICKNESS)

    text = label
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, DEBUG_FONT_SCALE, 1)
    tx = x0
    ty = max(15, y0 - 6)
    cv2.rectangle(img, (tx, ty - th - 6), (tx + tw + 6, ty + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (tx + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, DEBUG_FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)


# -------------------------
# Composite helpers
# -------------------------
def _pick_composite_pieces(tag_paths):
    """
    Select four tag pieces from TAG_DIR.

    If A/B/C/D names are available, they are used directly.
    Otherwise, the first four files in alphabetical order are used.
    """
    if len(tag_paths) < 4:
        raise ValueError(f"TAG_DIR must contain at least four tag pieces. Found: {len(tag_paths)}")

    stems = {os.path.splitext(os.path.basename(p))[0].lower(): p for p in tag_paths}
    # Prefer A/B/C/D files when available
    if all(k in stems for k in ["a", "b", "c", "d"]):
        chosen = [stems["a"], stems["b"], stems["c"], stems["d"]]
    else:
        chosen = sorted(tag_paths)[:4]

    # Sort names alphabetically
    chosen_sorted = sorted(chosen, key=lambda p: os.path.splitext(os.path.basename(p))[0].lower())
    names_sorted = [os.path.splitext(os.path.basename(p))[0] for p in chosen_sorted]
    return chosen_sorted, names_sorted


def build_composite_tag(tag_paths_4, piece_size=PIECE_SIZE_PX, gap=PIECE_GAP_PX):
    """
    Build a full-tag composite by arranging four pieces in a 2x2 layout.

    A separate binary mask is returned for each piece.
    Layout order: [0]=top-left, [1]=top-right, [2]=bottom-left, [3]=bottom-right.
    """
    # Size
    W = piece_size * 2 + gap
    H = piece_size * 2 + gap

    canvas = np.full((H, W, 3), 255, dtype=np.uint8)  # white background

    # Piece masks as uint8 0/255 arrays
    masks = [np.zeros((H, W), dtype=np.uint8) for _ in range(4)]

    positions = [
        (0, 0),                           # TL
        (piece_size + gap, 0),            # TR
        (0, piece_size + gap),            # BL
        (piece_size + gap, piece_size + gap),  # BR
    ]

    for i, (p, (x0, y0)) in enumerate(zip(tag_paths_4, positions)):
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Tag piece could not be read: {p}")
        # If the PNG has an alpha channel, composite it over a white background
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0
            rgb = img[:, :, :3].astype(np.float32)
            white = np.full_like(rgb, 255.0)
            rgb = rgb * alpha + white * (1.0 - alpha)
            img = np.clip(rgb, 0, 255).astype(np.uint8)

        if img.shape[0] != piece_size or img.shape[1] != piece_size:
            img = cv2.resize(img, (piece_size, piece_size), interpolation=cv2.INTER_AREA)

        canvas[y0:y0 + piece_size, x0:x0 + piece_size] = img
        masks[i][y0:y0 + piece_size, x0:x0 + piece_size] = 255

    return canvas, masks


def warp_and_crop_with_masks(img_bgr,
                             masks_u8,
                             yaw_deg=0.0,
                             pitch_deg=0.0,
                             roll_deg=0.0,
                             fov_deg=60.0,
                             dist_scale=3.0,
                             border_color=(255, 255, 255)):
    """
    Extended version of tag_embed_module.warp_and_crop_with_mask.

    The image and all provided masks are warped using the same homography.
    A tight crop is then applied using the union mask.

    Returns:
      tag_crop_bgr, union_mask01, piece_masks01(list)
    """
    h, w = img_bgr.shape[:2]

    f = 0.5 * w / np.tan(np.deg2rad(fov_deg / 2.0))
    K = np.array([[f, 0, w / 2.0],
                  [0, f, h / 2.0],
                  [0, 0, 1]], dtype=np.float32)

    half_w, half_h = w / 2.0, h / 2.0
    corners_3d = np.array([
        [-half_w, -half_h, 0],
        [ half_w, -half_h, 0],
        [ half_w,  half_h, 0],
        [-half_w,  half_h, 0],
    ], dtype=np.float32)

    yaw   = np.deg2rad(yaw_deg)
    pitch = np.deg2rad(pitch_deg)
    roll  = np.deg2rad(roll_deg)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch),  np.cos(pitch)]], dtype=np.float32)
    Ry = np.array([[ np.cos(yaw), 0, np.sin(yaw)],
                   [0,            1,          0],
                   [-np.sin(yaw), 0, np.cos(yaw)]], dtype=np.float32)
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll),  np.cos(roll), 0],
                   [0,             0,            1]], dtype=np.float32)

    R = Rz @ Ry @ Rx

    dist = max(w, h) * dist_scale
    t = np.array([[0.0], [0.0], [dist]], dtype=np.float32)

    pts_cam  = (R @ corners_3d.T) + t
    pts_proj = K @ pts_cam
    pts_proj /= pts_proj[2, :]
    dst = pts_proj[:2, :].T.astype(np.float32)

    src = np.array([
        [0,      0],
        [w - 1,  0],
        [w - 1,  h - 1],
        [0,      h - 1],
    ], dtype=np.float32)

    Hm = cv2.getPerspectiveTransform(src, dst)

    warped_img = cv2.warpPerspective(
        img_bgr, Hm, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_color
    )

    # Union mask: full ones (as in original) or union of given masks
    union_src = np.ones((h, w), dtype=np.uint8) * 255
    warped_union = cv2.warpPerspective(
        union_src, Hm, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    mask_bin = warped_union > 0
    if not np.any(mask_bin):
        # fallback: no crop
        tag_crop = warped_img
        union_mask01 = (warped_union > 0).astype(np.float32)
        piece_masks01 = [(cv2.warpPerspective(m, Hm, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0).astype(np.float32)
                         for m in masks_u8]
        return tag_crop, union_mask01, piece_masks01

    ys, xs = np.where(mask_bin)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    tag_crop = warped_img[y_min:y_max + 1, x_min:x_max + 1]
    union_crop = warped_union[y_min:y_max + 1, x_min:x_max + 1]
    union_mask01 = (union_crop > 0).astype(np.float32)

    piece_masks01 = []
    for m in masks_u8:
        wm = cv2.warpPerspective(
            m, Hm, (w, h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        wm_crop = wm[y_min:y_max + 1, x_min:x_max + 1]
        piece_masks01.append((wm_crop > 0).astype(np.float32))

    return tag_crop, union_mask01, piece_masks01


def main():
    global train_image_count, test_image_count, composite_train_image_count, composite_test_image_count, composite_train_tag_count, composite_test_tag_count

    ensure_dirs()
    scene_paths = list_images_in_dir(SCENE_DIR)
    tag_paths   = list_images_in_dir(TAG_DIR)

    if len(scene_paths) == 0:
        print("SCENE_DIR is empty. Exiting.")
        return
    if len(tag_paths) == 0:
        print("TAG_DIR is empty. Exiting.")
        return

    rng = np.random.default_rng()

    print(f"PIPELINE_MODE = {PIPELINE_MODE}")
    print(f"ALPHA_FORCE_METHOD = {ALPHA_FORCE_METHOD}")

    # ------------------------------------------------------------
    # 1) Individual sub-component generation: per-background object targets (S1-S4)
    # ------------------------------------------------------------
    target_obj_counts = {1: 120, 2: 60, 3: 40, 4: 40}
    global_image_idx = 0

    for scene_path in scene_paths:
        scene_path = str(scene_path)
        scene = cv2.imread(scene_path)
        if scene is None:
            print("Scene could not be read, skipping:", scene_path)
            continue

        base_name, ext = os.path.splitext(os.path.basename(scene_path))

        produced_obj_counts = {k: 0 for k in target_obj_counts.keys()}
        per_size_image_idx  = {k: 0 for k in target_obj_counts.keys()}  # 20% test split, every fifth image

        def all_targets_met_for_bg():
            return all(produced_obj_counts[k] >= target_obj_counts[k] for k in target_obj_counts)

        print("\n" + "=" * 60)
        print(f"BG: {base_name}{ext} | Targets: S1={target_obj_counts[1]}, S2={target_obj_counts[2]}, S3={target_obj_counts[3]}, S4={target_obj_counts[4]}")
        print("=" * 60)

        while not all_targets_met_for_bg():
            remaining = [k for k in target_obj_counts if produced_obj_counts[k] < target_obj_counts[k]]
            range_idx = int(rng.choice(remaining))
            min_size, max_size = SIZE_RANGES[range_idx - 1]

            global_image_idx += 1
            per_size_image_idx[range_idx] += 1

            if per_size_image_idx[range_idx] % 5 == 0:
                target_dir = TEST_DIR
                is_train = False
            else:
                target_dir = TRAIN_DIR
                is_train = True

            try:
                composed = random_crop(scene, CROP_SIZE, rng=rng)
            except ValueError as e:
                print(f"Warning: crop could not be extracted: {e} | scene={scene_path}")
                break

            layout = generate_random_layout(
                scene_shape=composed.shape,
                tag_paths=tag_paths,
                min_size=min_size,
                max_size=max_size,
                rng=rng,
            )

            objects = []

            for cfg in layout:
                try:
                    composed, bbox, alpha_method, blend_mode, occ_on, degradation = embed_tag_anti_overfitting(
                        composed,
                        cfg["tag_path"],
                        cfg["target_size_px"],
                        cfg["center_xy"],
                        cfg["yaw_deg"],
                        cfg["pitch_deg"],
                        cfg["roll_deg"],
                        cfg["fov_deg"],
                        cfg["dist_scale"],
                        rng=rng,
                        alpha_force=ALPHA_FORCE_METHOD,
                        pipeline_mode=PIPELINE_MODE
                    )
                    objects.append({"name": cfg["name"], "bbox": bbox})

                    if DEBUG_DRAW:
                        label = f"{cfg['name']} | {blend_mode}"
                        draw_debug_box(composed, bbox, label)

                except ValueError as e:
                    print(f"⚠️ Tag skipped: {e}")
                    continue

            if ENABLE_JPEG_COMPAT:
                qmin, qmax = JPEG_QUALITY_RANGE
                q = int(rng.integers(qmin, qmax + 1))
                composed = jpeg_encode_decode(composed, quality=q)

            suffix = f"_S{range_idx}_I{per_size_image_idx[range_idx]:05d}"
            img_name = base_name + suffix + ext
            xml_name = base_name + suffix + ".xml"

            img_path = os.path.join(target_dir, img_name)
            xml_path = os.path.join(target_dir, xml_name)

            cv2.imwrite(img_path, composed)
            write_voc_xml(img_path, composed.shape, objects, xml_path)

            if is_train:
                train_image_count += 1
            else:
                test_image_count += 1

            update_tag_counts(objects, is_train=is_train)
            produced_obj_counts[range_idx] += len(objects)

            print(
                f"[{base_name}] S{range_idx}: +{len(objects)} obj | "
                f"S1={produced_obj_counts[1]}/{target_obj_counts[1]}, "
                f"S2={produced_obj_counts[2]}/{target_obj_counts[2]}, "
                f"S3={produced_obj_counts[3]}/{target_obj_counts[3]}, "
                f"S4={produced_obj_counts[4]}/{target_obj_counts[4]} | "
                f"img_global={global_image_idx} ({'train' if is_train else 'test'})"
            )

    # ------------------------------------------------------------
    # 2) Full-tag composite generation: per-background targets
    #    This stage is added to the individual sub-component generation stage.
    #    For each background, COMPOSITE_TARGET_MAIN_TAGS full-tag composites are generated.
    #    XML annotations store the four pieces inside each composite as separate objects.
    # ------------------------------------------------------------
    composite_paths_4, composite_names_4 = _pick_composite_pieces(tag_paths)

    # Aggregate total targets by background for reporting
    num_valid_bgs_for_composite = 0
    total_main_target_all = 0
    total_main_made_all = 0

    # Track expected and generated sub-component boxes
    composite_expected_extra_all = {name: 0 for name in composite_names_4}

    print("\n" + "=" * 60)
    print("ADDITIONAL STAGE: FULL-TAG COMPOSITE GENERATION PER BACKGROUND")
    print("Pieces:", composite_names_4)
    print("Target full-tag composite counts per BG:", COMPOSITE_TARGET_MAIN_TAGS)
    print("=" * 60)

    # Fill separate targets for each background
    for comp_scene_path in scene_paths:
        comp_scene_path = str(comp_scene_path)
        scene = cv2.imread(comp_scene_path)
        if scene is None:
            print("Scene could not be read for composite generation, skipping:", comp_scene_path)
            continue

        base_name, ext = os.path.splitext(os.path.basename(comp_scene_path))
        num_valid_bgs_for_composite += 1

        # Per-background counters
        per_size_comp_img_idx = {k: 0 for k in COMPOSITE_TARGET_MAIN_TAGS.keys()}  # 20% test split, every fifth image
        made_main_tags = {k: 0 for k in COMPOSITE_TARGET_MAIN_TAGS.keys()}
        fail_tries = {k: 0 for k in COMPOSITE_TARGET_MAIN_TAGS.keys()}

        total_main_target_bg = sum(COMPOSITE_TARGET_MAIN_TAGS.values())
        total_main_target_all += total_main_target_bg

        def composite_done_bg():
            return all(made_main_tags[k] >= COMPOSITE_TARGET_MAIN_TAGS[k] for k in COMPOSITE_TARGET_MAIN_TAGS)

        print(f"\n[COMPOSITE-BG] {base_name}{ext} | target: {COMPOSITE_TARGET_MAIN_TAGS}")

        while not composite_done_bg():
            # Select a size range with remaining targets and available retry budget
            remaining = [k for k in COMPOSITE_TARGET_MAIN_TAGS if made_main_tags[k] < COMPOSITE_TARGET_MAIN_TAGS[k]]
            remaining = [k for k in remaining if fail_tries[k] < MAX_TRIES_PER_RANGE]
            if not remaining:
                print(f"[WARN] {base_name}: Some size ranges could not fit in this BG within the retry limit. Skipping BG.")
                break

            range_idx = int(rng.choice(remaining))
            min_size, max_size = COMPOSITE_SIZE_RANGES[range_idx - 1]

            # Determine how many full-tag composites to try in this iteration
            need = COMPOSITE_TARGET_MAIN_TAGS[range_idx] - made_main_tags[range_idx]
            max_tags = int(min(COMPOSITE_MAX_TAGS_PER_IMAGE, max(1, need)))
            min_tags = int(min(COMPOSITE_MIN_TAGS_PER_IMAGE, max_tags))

            # Extract a separate random BG patch for each generated image
            try:
                composed = random_crop(scene, CROP_SIZE, rng=rng)
            except ValueError as e:
                print(f"Warning: composite crop could not be extracted: {e} | scene={comp_scene_path}")
                fail_tries[range_idx] += 1
                continue

            # Build the full-tag composite
            composite_img, piece_masks_u8 = build_composite_tag(composite_paths_4, piece_size=PIECE_SIZE_PX, gap=PIECE_GAP_PX)

            # Layout for placing multiple full-tag composites in the same image
            layout = generate_random_layout(
                scene_shape=composed.shape,
                tag_paths=["__COMPOSITE__"],
                min_size=min_size,
                max_size=max_size,
                min_tags=min_tags,
                max_tags=max_tags,
                rng=rng
            )

            objects = []
            placed_main_tags = 0

            for cfg in layout:
                target_size_px = int(cfg["target_size_px"])
                cx, cy = cfg["center_xy"]
                yaw_deg = cfg["yaw_deg"]
                pitch_deg = cfg["pitch_deg"]
                roll_deg = cfg["roll_deg"]
                fov_deg = cfg["fov_deg"]
                dist_scale = cfg["dist_scale"]

                # Resize the full-tag composite to the target size
                composite_resized = cv2.resize(
                    composite_img, (target_size_px, target_size_px), interpolation=cv2.INTER_AREA
                )

                # Resize masks to the same target size
                resized_masks_u8 = []
                for m_u8 in piece_masks_u8:
                    m_rs = cv2.resize(m_u8, (target_size_px, target_size_px), interpolation=cv2.INTER_NEAREST)
                    resized_masks_u8.append(m_rs)

                # Warp and crop the full-tag composite and piece masks together
                try:
                    tag_crop_bgr, union_mask01, piece_masks01 = warp_and_crop_with_masks(
                        composite_resized,
                        resized_masks_u8,
                        yaw_deg=yaw_deg,
                        pitch_deg=pitch_deg,
                        roll_deg=roll_deg,
                        fov_deg=fov_deg,
                        dist_scale=dist_scale,
                        border_color=(255, 255, 255),
                    )
                except Exception:
                    continue

                # Quick boundary check for placement
                h_tag, w_tag = tag_crop_bgr.shape[:2]
                if (cx - w_tag / 2) < 0 or (cy - h_tag / 2) < 0 or (cx + w_tag / 2) >= CROP_SIZE or (cy + h_tag / 2) >= CROP_SIZE:
                    continue

                # Overlay using the union mask of the full-tag composite
                try:
                    composed, bbox_union, meta = anti_overfitting_overlay(
                        composed,
                        tag_crop_bgr,
                        union_mask01,
                        center_xy=(int(cx), int(cy)),
                        perspective_angles=(yaw_deg, pitch_deg),
                        rng=rng,
                        alpha_force=ALPHA_FORCE_METHOD,
                        pipeline_mode=PIPELINE_MODE
                    )
                except ValueError:
                    continue

                # Union ROI coordinates
                x0 = int(cx - w_tag / 2)
                y0 = int(cy - h_tag / 2)

                # Compute and add sub-component bounding boxes
                added_any = False
                for name, pm01 in zip(composite_names_4, piece_masks01):
                    ys, xs = np.where(pm01 > 0.5)
                    if len(xs) == 0 or len(ys) == 0:
                        continue
                    xmin = int(x0 + xs.min())
                    xmax = int(x0 + xs.max())
                    ymin = int(y0 + ys.min())
                    ymax = int(y0 + ys.max())

                    xmin = max(0, min(CROP_SIZE - 1, xmin))
                    ymin = max(0, min(CROP_SIZE - 1, ymin))
                    xmax = max(0, min(CROP_SIZE - 1, xmax))
                    ymax = max(0, min(CROP_SIZE - 1, ymax))
                    if xmax <= xmin or ymax <= ymin:
                        continue
                    objects.append({"name": name, "bbox": (xmin, ymin, xmax, ymax)})
                    added_any = True

                if added_any:
                    placed_main_tags += 1

            # Count as a failed attempt if no composite was placed
            if placed_main_tags == 0 or len(objects) == 0:
                fail_tries[range_idx] += 1
                continue

            # Train/test split by size range: 20% test, every fifth image
            per_size_comp_img_idx[range_idx] += 1
            is_train = (per_size_comp_img_idx[range_idx] % 5 != 0)
            target_dir = TRAIN_DIR if is_train else TEST_DIR

            # Composite counters for images and boxes
            if is_train:
                composite_train_image_count += 1
            else:
                composite_test_image_count += 1

            for obj in objects:
                nm = obj["name"]
                if is_train:
                    composite_train_tag_count[nm] = composite_train_tag_count.get(nm, 0) + 1
                else:
                    composite_test_tag_count[nm] = composite_test_tag_count.get(nm, 0) + 1

            # JPEG compatibility
            if ENABLE_JPEG_COMPAT:
                qmin, qmax = JPEG_QUALITY_RANGE
                q = int(rng.integers(qmin, qmax + 1))
                composed = jpeg_encode_decode(composed, quality=q)

            # File name based on the background
            suffix = f"_COMP_S{range_idx}_I{per_size_comp_img_idx[range_idx]:04d}"
            img_name = f"{base_name}{suffix}.jpg"
            xml_name = f"{base_name}{suffix}.xml"

            img_path = os.path.join(target_dir, img_name)
            xml_path = os.path.join(target_dir, xml_name)

            cv2.imwrite(img_path, composed)
            write_voc_xml(img_path, composed.shape, objects, xml_path)

            update_tag_counts(objects, is_train=is_train)

            # Per-background target counter for full-tag composites
            made_main_tags[range_idx] += placed_main_tags
            total_main_made_all += placed_main_tags

            # Expected sub-component box contribution
            for nm in composite_names_4:
                composite_expected_extra_all[nm] += placed_main_tags

            print(
                f"[COMPOSITE] {base_name} S{range_idx}: +{placed_main_tags} main-tag ({4*placed_main_tags} obj) | "
                f"S1={made_main_tags[1]}/{COMPOSITE_TARGET_MAIN_TAGS[1]}, "
                f"S2={made_main_tags[2]}/{COMPOSITE_TARGET_MAIN_TAGS[2]}, "
                f"S3={made_main_tags[3]}/{COMPOSITE_TARGET_MAIN_TAGS[3]}, "
                f"S4={made_main_tags[4]}/{COMPOSITE_TARGET_MAIN_TAGS[4]} | "
                f"({'train' if is_train else 'test'})"
            )
    # ------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------
    print("\n==============================")
    print("      TRAIN REPORT")
    print("==============================")
    total_train_images = train_image_count + composite_train_image_count
    print(f"Total train images (individual sub-component stage): {train_image_count}")
    print(f"Total train images (full-tag composite stage): {composite_train_image_count}")
    print(f"Total train images (overall): {total_train_images}")

    # Class-wise box counts: individual stage + composite stage + total
    all_train_names = sorted(set(train_tag_count.keys()) | set(composite_train_tag_count.keys()))
    for name in all_train_names:
        base_c = train_tag_count.get(name, 0)
        comp_c = composite_train_tag_count.get(name, 0)
        print(f"{name}: {base_c + comp_c} bbox  (individual={base_c}, composite={comp_c})")

    print("\n==============================")
    print("       TEST REPORT")
    print("==============================")
    total_test_images = test_image_count + composite_test_image_count
    print(f"Total test images (individual sub-component stage): {test_image_count}")
    print(f"Total test images (full-tag composite stage): {composite_test_image_count}")
    print(f"Total test images (overall): {total_test_images}")

    all_test_names = sorted(set(test_tag_count.keys()) | set(composite_test_tag_count.keys()))
    for name in all_test_names:
        base_c = test_tag_count.get(name, 0)
        comp_c = composite_test_tag_count.get(name, 0)
        print(f"{name}: {base_c + comp_c} bbox  (individual={base_c}, composite={comp_c})")

    # Additional summary for the full-tag composite stage
    print("\n==============================")
    print("  FULL-TAG COMPOSITE SUMMARY")
    print("==============================")
    print(f"Valid BG count: {num_valid_bgs_for_composite}")
    print(f"Target full-tag composites per BG: {sum(COMPOSITE_TARGET_MAIN_TAGS.values())}")
    print(f"Target full-tag composites (total): {total_main_target_all} | Generated full-tag composites (total): {total_main_made_all}")
    print(f"Full-tag composite images: train={composite_train_image_count}, test={composite_test_image_count}, total={composite_train_image_count + composite_test_image_count}")
    # Sub-component box contribution
    for nm in composite_names_4:
        got = composite_train_tag_count.get(nm, 0) + composite_test_tag_count.get(nm, 0)
        print(f"{nm}: target +{total_main_target_all} bbox | generated +{got} bbox")

    # CSV reports: overall totals from individual and composite stages
    merged_train = {}
    for k in set(train_tag_count.keys()) | set(composite_train_tag_count.keys()):
        merged_train[k] = train_tag_count.get(k, 0) + composite_train_tag_count.get(k, 0)
    merged_test = {}
    for k in set(test_tag_count.keys()) | set(composite_test_tag_count.keys()):
        merged_test[k] = test_tag_count.get(k, 0) + composite_test_tag_count.get(k, 0)

    save_report_csv(os.path.join(OUTPUT_DIR, "train_report.csv"), total_train_images, merged_train)
    save_report_csv(os.path.join(OUTPUT_DIR, "test_report.csv"), total_test_images, merged_test)

    # Optional composite-only reports
    save_report_csv(os.path.join(OUTPUT_DIR, "train_report_composite_only.csv"), composite_train_image_count, composite_train_tag_count)
    save_report_csv(os.path.join(OUTPUT_DIR, "test_report_composite_only.csv"), composite_test_image_count, composite_test_tag_count)

if __name__ == "__main__":
    main()
