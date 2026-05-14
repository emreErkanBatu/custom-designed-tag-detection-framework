# -*- coding: utf-8 -*-
"""
Synthetic data generation for CNN-1, the ROI/full-tag detector.
----------------------------------------------------------------
This script generates composite full-tag samples for ROI detector training.

Main settings:
- Piece-wise tag placement is not used in this script.
- A single composite full tag is placed on background patches using the same randomization strategy.
- Each XML file uses one class name, "ABCD", with one full-tag bounding box per placed tag.
- COMPOSITE_SIZE_RANGES:
    (80,150), (150,300), (300,350), (350,400)
- Target full-tag counts for each background image:
    S1:120, S2:60, S3:60, S4:60
- Train/test split: 80% / 20%, applied per size range by assigning every fifth image to the test set.
- A maximum retry limit is used for each size range to avoid endless loops.

Note:
- The full tag is built from four pieces in TAG_DIR, arranged alphabetically in a 2x2 layout on a white background.
- If placement constraints allow it, multiple full tags can be placed in the same image.
"""

import os
import cv2
import csv
import xml.etree.ElementTree as ET
import numpy as np

from tag_layout_module import generate_random_layout
from anti_overfitting_tag_embed import anti_overfitting_overlay
from tag_embed_module import warp_and_crop_with_mask


# -------------------------
# IO
# -------------------------
SCENE_DIR = "scenes"
TAG_DIR   = "tags"

OUTPUT_DIR =  r"E:\TFOD\datasets\05_tag_part\images"
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
CROP_SIZE = 640

# CNN-1 class label for the full tag
CNN1_CLASS_NAME = "ABCD"


# -------------------------
# CNN-1 size ranges & targets (per BG)
# -------------------------
COMPOSITE_SIZE_RANGES = [
    (80, 150),  # S1
    (150, 300),  # S2
    (300, 350),  # S3
    (350, 400),  # S4
]

COMPOSITE_TARGET_MAIN_TAGS_PER_BG = {1: 120, 2: 60, 3: 60, 4: 60}

# Allow multiple full tags in the same image
COMPOSITE_MIN_TAGS_PER_IMAGE = 1
COMPOSITE_MAX_TAGS_PER_IMAGE = 20

# Loop-safety limit
MAX_TRIES_PER_RANGE = 250  # Failed-attempt limit per background and size range


# -------------------------
# Composite tag geometry
# -------------------------
PIECE_SIZE_PX = 500
PIECE_GAP_PX  = 36


# -------------------------
# Helpers
# -------------------------
def ensure_dirs():
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)


def list_images_in_dir(directory, exts=(".png", ".jpg", ".jpeg", ".bmp")):
    files = []
    if os.path.exists(directory):
        for fname in os.listdir(directory):
            if fname.lower().endswith(exts):
                files.append(os.path.join(directory, fname))
    return sorted(files)


def random_crop(scene_bgr: np.ndarray, crop_size: int, rng: np.random.Generator) -> np.ndarray:
    h, w = scene_bgr.shape[:2]
    if h < crop_size or w < crop_size:
        raise ValueError(f"Scene too small for {crop_size}x{crop_size} crop: {w}x{h}")
    x0 = int(rng.integers(0, w - crop_size + 1))
    y0 = int(rng.integers(0, h - crop_size + 1))
    return scene_bgr[y0:y0 + crop_size, x0:x0 + crop_size].copy()


def jpeg_encode_decode(bgr: np.ndarray, quality: int) -> np.ndarray:
    quality = int(np.clip(quality, 30, 100))
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return bgr
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return dec if dec is not None else bgr


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


def save_report_csv(csv_path, image_count, tag_count_dict):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Total Images", image_count])
        writer.writerow([])
        writer.writerow(["Class Name", "Total BBox Count"])
        for tag, c in sorted(tag_count_dict.items()):
            writer.writerow([tag, c])
    print(f"CSV saved -> {csv_path}")


def _pick_composite_pieces(tag_paths):
    """
    Select four tag pieces from TAG_DIR:
    - use a, b, c, d when available;
    - otherwise use the first four files in alphabetical order.
    """
    if len(tag_paths) < 4:
        raise ValueError(f"TAG_DIR must contain at least four tag pieces. Found: {len(tag_paths)}")

    stems = {os.path.splitext(os.path.basename(p))[0].lower(): p for p in tag_paths}
    if all(k in stems for k in ["a", "b", "c", "d"]):
        chosen = [stems["a"], stems["b"], stems["c"], stems["d"]]
    else:
        chosen = sorted(tag_paths)[:4]

    chosen_sorted = sorted(chosen, key=lambda p: os.path.splitext(os.path.basename(p))[0].lower())
    return chosen_sorted


def build_composite_tag(tag_paths_4, piece_size=PIECE_SIZE_PX, gap=PIECE_GAP_PX):
    """
    Build one full tag by arranging four pieces in a 2x2 layout on a white background.
    If an alpha channel exists, the piece is composited onto the white background.
    """
    W = piece_size * 2 + gap
    H = piece_size * 2 + gap

    canvas = np.full((H, W, 3), 255, dtype=np.uint8)  # white bg

    positions = [
        (0, 0),                           # TL
        (piece_size + gap, 0),            # TR
        (0, piece_size + gap),            # BL
        (piece_size + gap, piece_size + gap),  # BR
    ]

    for p, (x0, y0) in zip(tag_paths_4, positions):
        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Tag piece could not be read: {p}")

        # If RGBA, composite it onto a white background.
        if img.ndim == 3 and img.shape[2] == 4:
            alpha = img[:, :, 3:4].astype(np.float32) / 255.0
            rgb = img[:, :, :3].astype(np.float32)
            white = np.full_like(rgb, 255.0)
            rgb = rgb * alpha + white * (1.0 - alpha)
            img = np.clip(rgb, 0, 255).astype(np.uint8)

        if img.shape[0] != piece_size or img.shape[1] != piece_size:
            img = cv2.resize(img, (piece_size, piece_size), interpolation=cv2.INTER_AREA)

        canvas[y0:y0 + piece_size, x0:x0 + piece_size] = img

    return canvas


def draw_debug_box(img, bbox, label: str):
    x0, y0, x1, y1 = [int(v) for v in bbox]
    cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 255), DEBUG_THICKNESS)

    text = label
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, DEBUG_FONT_SCALE, 1)
    tx = x0
    ty = max(15, y0 - 6)
    cv2.rectangle(img, (tx, ty - th - 6), (tx + tw + 6, ty + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (tx + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, DEBUG_FONT_SCALE, (255, 255, 255), 1, cv2.LINE_AA)


def main():
    ensure_dirs()
    scene_paths = list_images_in_dir(SCENE_DIR)
    tag_paths   = list_images_in_dir(TAG_DIR)

    if len(scene_paths) == 0:
        print("SCENE_DIR is empty. Exiting.")
        return
    if len(tag_paths) < 4:
        print("TAG_DIR must contain at least four tag pieces. Exiting.")
        return

    rng = np.random.default_rng()

    print(f"PIPELINE_MODE = {PIPELINE_MODE}")
    print(f"ALPHA_FORCE_METHOD = {ALPHA_FORCE_METHOD}")

    # Build the white-background composite tag once.
    piece_paths_4 = _pick_composite_pieces(tag_paths)
    composite_img = build_composite_tag(piece_paths_4, piece_size=PIECE_SIZE_PX, gap=PIECE_GAP_PX)

    # Global report counters
    train_image_count = 0
    test_image_count  = 0
    train_tag_count = {CNN1_CLASS_NAME: 0}
    test_tag_count  = {CNN1_CLASS_NAME: 0}

    # Generate samples for each background image.
    for scene_path in scene_paths:
        scene = cv2.imread(scene_path)
        if scene is None:
            print("Scene could not be read, skipping:", scene_path)
            continue

        base_name, ext = os.path.splitext(os.path.basename(scene_path))

        produced_main = {k: 0 for k in COMPOSITE_TARGET_MAIN_TAGS_PER_BG.keys()}
        per_size_img_idx = {k: 0 for k in COMPOSITE_TARGET_MAIN_TAGS_PER_BG.keys()}  # 20% test split: every fifth image
        fail_tries = {k: 0 for k in COMPOSITE_TARGET_MAIN_TAGS_PER_BG.keys()}

        def done_for_bg():
            return all(produced_main[k] >= COMPOSITE_TARGET_MAIN_TAGS_PER_BG[k] for k in COMPOSITE_TARGET_MAIN_TAGS_PER_BG)

        print("\n" + "=" * 70)
        print(f"BG: {base_name}{ext} | Target full tags: S1=120, S2=60, S3=60, S4=60")
        print("=" * 70)

        # Continue until the targets for this background are reached.
        while not done_for_bg():
            remaining = [k for k in COMPOSITE_TARGET_MAIN_TAGS_PER_BG if produced_main[k] < COMPOSITE_TARGET_MAIN_TAGS_PER_BG[k]]
            # Remove a size range from the remaining list if it reaches the failure limit.
            remaining = [k for k in remaining if fail_tries[k] < MAX_TRIES_PER_RANGE]
            if not remaining:
                print(f"[WARN] {base_name}: Some sizes did not fit this background within the failure limit. Skipping this background.")
                break

            range_idx = int(rng.choice(remaining))
            min_size, max_size = COMPOSITE_SIZE_RANGES[range_idx - 1]

            # Decide how many full tags to place in this iteration.
            need = COMPOSITE_TARGET_MAIN_TAGS_PER_BG[range_idx] - produced_main[range_idx]
            max_tags = int(min(COMPOSITE_MAX_TAGS_PER_IMAGE, max(1, need)))
            min_tags = int(min(COMPOSITE_MIN_TAGS_PER_IMAGE, max_tags))

            # BG patch
            try:
                composed = random_crop(scene, CROP_SIZE, rng=rng)
            except ValueError:
                fail_tries[range_idx] += 1
                continue

            # Generate non-overlapping positions and scales.
            layout = generate_random_layout(
                scene_shape=composed.shape,
                tag_paths=["__COMPOSITE__"],  # dummy
                min_size=min_size,
                max_size=max_size,
                min_tags=min_tags,
                max_tags=max_tags,
                rng=rng
            )

            objects = []
            placed = 0

            for cfg in layout:
                target_size_px = int(cfg["target_size_px"])
                cx, cy = cfg["center_xy"]

                # Resize the composite tag to the target square size.
                comp_rs = cv2.resize(composite_img, (target_size_px, target_size_px), interpolation=cv2.INTER_AREA)

                # Warp and crop; the mask represents the occupied tag area.
                try:
                    tag_crop_bgr, mask_crop = warp_and_crop_with_mask(
                        comp_rs,
                        yaw_deg=cfg["yaw_deg"],
                        pitch_deg=cfg["pitch_deg"],
                        roll_deg=cfg["roll_deg"],
                        fov_deg=cfg["fov_deg"],
                        dist_scale=cfg["dist_scale"],
                        border_color=(255, 255, 255)
                    )
                except Exception:
                    continue

                h_tag, w_tag = tag_crop_bgr.shape[:2]

                # Check image boundaries before overlay.
                if (cx - w_tag / 2) < 0 or (cy - h_tag / 2) < 0 or (cx + w_tag / 2) >= CROP_SIZE or (cy + h_tag / 2) >= CROP_SIZE:
                    continue

                # Overlay + bbox
                try:
                    composed, bbox_union, _meta = anti_overfitting_overlay(
                        composed,
                        tag_crop_bgr,
                        mask_crop,
                        center_xy=(int(cx), int(cy)),
                        perspective_angles=(cfg["yaw_deg"], cfg["pitch_deg"]),
                        rng=rng,
                        alpha_force=ALPHA_FORCE_METHOD,
                        pipeline_mode=PIPELINE_MODE
                    )
                except ValueError:
                    continue

                # DEBUG draw (bbox_union) on composed image
                if DEBUG_DRAW and bbox_union is not None:
                    try:
                        # If ALPHA_FORCE_METHOD == "random", _meta["alpha"] holds the ACTUAL chosen method+params (e.g., "dist(...)")
                        alpha_dbg = None
                        if isinstance(_meta, dict):
                            alpha_dbg = _meta.get("alpha", None)
                        if not alpha_dbg:
                            alpha_dbg = ALPHA_FORCE_METHOD
                        dbg_label = f"{CNN1_CLASS_NAME} | {alpha_dbg}"
                        draw_debug_box(composed, bbox_union, dbg_label)
                    except Exception:
                        pass

                objects.append({"name": CNN1_CLASS_NAME, "bbox": bbox_union})
                placed += 1

                # Stop if the target count has been reached.
                if produced_main[range_idx] + placed >= COMPOSITE_TARGET_MAIN_TAGS_PER_BG[range_idx]:
                    break

            if placed == 0:
                fail_tries[range_idx] += 1
                continue

            # Train/test split by size range
            per_size_img_idx[range_idx] += 1
            is_train = (per_size_img_idx[range_idx] % 5 != 0)
            target_dir = TRAIN_DIR if is_train else TEST_DIR

            # JPEG compatibility
            if ENABLE_JPEG_COMPAT:
                qmin, qmax = JPEG_QUALITY_RANGE
                q = int(rng.integers(qmin, qmax + 1))
                composed = jpeg_encode_decode(composed, quality=q)

            # Save outputs
            suffix = f"_ROI_S{range_idx}_I{per_size_img_idx[range_idx]:05d}"
            img_name = f"{base_name}{suffix}.jpg"
            xml_name = f"{base_name}{suffix}.xml"

            img_path = os.path.join(target_dir, img_name)
            xml_path = os.path.join(target_dir, xml_name)

            cv2.imwrite(img_path, composed)
            write_voc_xml(img_path, composed.shape, objects, xml_path)

            # Counters
            produced_main[range_idx] += placed
            if is_train:
                train_image_count += 1
                train_tag_count[CNN1_CLASS_NAME] += placed
            else:
                test_image_count += 1
                test_tag_count[CNN1_CLASS_NAME] += placed

            # Log
            print(
                f"[{base_name}] S{range_idx}: +{placed} ABCD | "
                f"S1={produced_main[1]}/{COMPOSITE_TARGET_MAIN_TAGS_PER_BG[1]}, "
                f"S2={produced_main[2]}/{COMPOSITE_TARGET_MAIN_TAGS_PER_BG[2]}, "
                f"S3={produced_main[3]}/{COMPOSITE_TARGET_MAIN_TAGS_PER_BG[3]}, "
                f"S4={produced_main[4]}/{COMPOSITE_TARGET_MAIN_TAGS_PER_BG[4]} | "
                f"{'train' if is_train else 'test'}"
            )

        # Background-level summary
        print(f"[DONE] {base_name}: S1={produced_main[1]}, S2={produced_main[2]}, S3={produced_main[3]}, S4={produced_main[4]}")

    # -------------------------
    # REPORT
    # -------------------------
    print("\n==============================")
    print("      TRAIN REPORT")
    print("==============================")
    print(f"Total Train Images: {train_image_count}")
    print(f"{CNN1_CLASS_NAME}: {train_tag_count.get(CNN1_CLASS_NAME, 0)} bbox")

    print("\n==============================")
    print("       TEST REPORT")
    print("==============================")
    print(f"Total Test Images: {test_image_count}")
    print(f"{CNN1_CLASS_NAME}: {test_tag_count.get(CNN1_CLASS_NAME, 0)} bbox")

    # CSV
    save_report_csv(os.path.join(OUTPUT_DIR, "train_report.csv"), train_image_count, train_tag_count)
    save_report_csv(os.path.join(OUTPUT_DIR, "test_report.csv"), test_image_count, test_tag_count)


if __name__ == "__main__":
    main()
