
# -*- coding: utf-8 -*-
"""
two_stage_infer_class.py

Two-stage TensorFlow inference pipeline for comparing the previous and proposed
detection models.

Main behavior:
- infer() takes a BGR numpy image directly.
- ROI and sub-component detectors use separate confidence thresholds.
- If the ROI score is below the ROI threshold, the corresponding sub-component
  detector is not executed and the sub-component scores are set to 0.0.
- noROI.jpg is used as the fallback ROI image when no valid ROI is detected.
- The result object provides ROI images and score dictionaries through attribute
  access and dict-like get() access.

Notes:
- Label names are parsed from label_map.pbtxt and used as score keys.
- Output images contain only colored bounding boxes.
- Unicode-safe OpenCV image loading is used for noROI.jpg.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import cv2
import numpy as np
import tensorflow as tf


# ---------------------------
# Unicode-safe OpenCV IO
# ---------------------------

def _imread_unicode(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


# ---------------------------
# Label map parser (pbtxt)
# supports name/display_name with single or double quotes
# ---------------------------

_ITEM_RE = re.compile(r"item\s*\{.*?\}", flags=re.S)
_ID_RE = re.compile(r"\bid\s*:\s*(\d+)")
_NAME_RE = re.compile(r"\bname\s*:\s*['\"]([^'\"]+)['\"]")
_DNAME_RE = re.compile(r"\bdisplay_name\s*:\s*['\"]([^'\"]+)['\"]")


def _parse_label_map_pbtxt(pbtxt_path: str) -> Dict[int, str]:
    if not pbtxt_path or not os.path.exists(pbtxt_path):
        return {}
    txt = open(pbtxt_path, "r", encoding="utf-8", errors="ignore").read()
    items = _ITEM_RE.findall(txt)
    out: Dict[int, str] = {}
    for it in items:
        mid = _ID_RE.search(it)
        if not mid:
            continue
        cid = int(mid.group(1))
        d = _DNAME_RE.search(it)
        n = _NAME_RE.search(it)
        name = (d.group(1).strip() if d else (n.group(1).strip() if n else str(cid)))
        out[cid] = name
    return out


# ---------------------------
# Deterministic colors
# ---------------------------

def _hsv_color_from_id(class_id: int) -> Tuple[int, int, int]:
    h = int((class_id * 37) % 180)
    s = 220
    v = 255
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ---------------------------
# TF SavedModel helpers
# ---------------------------

def _enable_memory_growth() -> None:
    try:
        gpus = tf.config.list_physical_devices("GPU")
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
    except Exception:
        pass


class _TFODDetector:
    def __init__(self, saved_model_dir: str):
        if not os.path.exists(saved_model_dir):
            raise FileNotFoundError(f"SavedModel folder not found: {saved_model_dir}")
        self.model = tf.saved_model.load(saved_model_dir)
        self.fn = self.model.signatures.get("serving_default")
        if self.fn is None:
            self.fn = list(self.model.signatures.values())[0]

    def infer(self, image_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        inp = tf.convert_to_tensor(rgb, dtype=tf.uint8)[tf.newaxis, ...]
        out = self.fn(inp)

        boxes = out.get("detection_boxes")
        scores = out.get("detection_scores")
        classes = out.get("detection_classes")

        if boxes is None or scores is None or classes is None:
            raise KeyError("Detector output missing one of: detection_boxes, detection_scores, detection_classes")

        return boxes.numpy()[0], scores.numpy()[0], classes.numpy()[0]


# ---------------------------
# ROI helpers
# ---------------------------

def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _compute_roi_from_box(box_norm: np.ndarray, W: int, H: int, roi_x: float) -> Tuple[int, int, int, int]:
    """Expand bbox by 1/roi_x around center (roi_x < 1 expands)."""
    y1n, x1n, y2n, x2n = [float(v) for v in box_norm.tolist()]
    x1 = int(round(x1n * W))
    y1 = int(round(y1n * H))
    x2 = int(round(x2n * W))
    y2 = int(round(y2n * H))

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    roi_w = int(round(bw / max(1e-6, float(roi_x))))
    roi_h = int(round(bh / max(1e-6, float(roi_x))))

    pad_x = max(0, (roi_w - bw) // 2)
    pad_y = max(0, (roi_h - bh) // 2)

    rx1 = _clamp(x1 - pad_x, 0, W - 1)
    ry1 = _clamp(y1 - pad_y, 0, H - 1)
    rx2 = _clamp(x2 + pad_x, rx1 + 1, W)
    ry2 = _clamp(y2 + pad_y, ry1 + 1, H)
    return rx1, ry1, rx2, ry2


def _resize_keep_aspect_long_side(img_bgr: np.ndarray, long_side: int) -> Tuple[np.ndarray, float]:
    H, W = img_bgr.shape[:2]
    cur = max(H, W)
    if long_side <= 0 or cur == long_side:
        return img_bgr, 1.0
    scale = float(long_side) / float(cur)
    newW = max(1, int(round(W * scale)))
    newH = max(1, int(round(H * scale)))
    resized = cv2.resize(img_bgr, (newW, newH), interpolation=cv2.INTER_LINEAR)
    return resized, float(scale)


# ---------------------------
# Drawing + scoring
# ---------------------------

def _draw_boxes_only(image_bgr, boxes, scores, classes, score_thresh, thickness=None):
    """
    Draw only the highest-scoring bbox for each class.

    - If thickness is None, it is auto-scaled based on ROI long side (same behavior as before).
    - score_thresh is used ONLY for drawing (logging uses raw scores elsewhere).
    """
    if boxes is None or scores is None or classes is None:
        return
    if len(boxes) == 0:
        return

    H, W = image_bgr.shape[:2]
    long_side = max(H, W)

    # Auto thickness (if not explicitly provided)
    if thickness is None:
        if long_side < 50:
            t = 1
        else:
            # 50-149 => 2, 150-249 => 3, 250-349 => 4, ...
            t = 2 + (long_side - 50) // 100
        thickness = int(max(1, t))

    # Best bbox index per class id (pick highest score)
    best_idx = {}
    for i, (sc, cls) in enumerate(zip(scores, classes)):
        scf = float(sc)
        cid = int(cls)
        if (cid not in best_idx) or (scf > float(scores[best_idx[cid]])):
            best_idx[cid] = i

    for cid, i in best_idx.items():
        if float(scores[i]) < float(score_thresh):
            continue
        y1n, x1n, y2n, x2n = [float(v) for v in boxes[i].tolist()]
        x1 = int(round(x1n * W))
        y1 = int(round(y1n * H))
        x2 = int(round(x2n * W))
        y2 = int(round(y2n * H))
        color = _hsv_color_from_id(cid)
        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, int(thickness))



def _collect_max_scores(scores: np.ndarray,
                        classes: np.ndarray,
                        id_to_name: Dict[int, str],
                        score_thresh: float,
                        fill_zero_for_all_labels: bool = False) -> Dict[str, float]:
    """
    Returns label->max_score for each label present in detections.

    IMPORTANT (objective logging):
    - `score_thresh` is NOT used to filter scores here anymore. Even if a part score is below
      the threshold, it will still be reported in the panel and written to CSV.
    - Thresholding is applied ONLY for drawing bboxes and for decision logic (e.g., temporal K=3).
    - If fill_zero_for_all_labels=True, ensures EVERY label in id_to_name exists and defaults to 0.0.
    """
    out: Dict[str, float] = {}
    for sc, cls in zip(scores, classes):
        scf = float(sc)
        name = id_to_name.get(int(cls), str(int(cls)))
        if (name not in out) or (scf > out[name]):
            out[name] = scf

    if fill_zero_for_all_labels:
        for _, name in id_to_name.items():
            if name not in out:
                out[name] = 0.0

    return out
# ---------------------------
# Public API
# ---------------------------

@dataclass
class TwoStageConfig:
    model1_savedmodel_dir: str
    model2_savedmodel_dir: str
    model3_savedmodel_dir: str
    model4_savedmodel_dir: str

    label_map_path_model1: str
    label_map_path_model2: str
    label_map_path_model3: str
    label_map_path_model4: str

    score_thresh_roi: float = 0.40   # for M1 and M3
    score_thresh_tag: float = 0.10   # for M2 and M4

    roi_x: float = 0.60
    roi_x_temporal_mul: float = 1.20  # when M3 is skipped (temporal ROI), use roi_x * mul (capped at 1.0)

    roi2_long_side: int = 512

    box_thickness: int = 2


class InferResult:
    """Small result container: attribute access for roi images + dict-like get()."""
    def __init__(self, data: Dict[str, Any], roi1: np.ndarray, roi2: np.ndarray):
        self._data = data
        self.roi1 = roi1
        self.roi2 = roi2

    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def __getitem__(self, key: str):
        return self._data[key]

    def as_dict(self) -> Dict[str, Any]:
        return dict(self._data)


class TwoStageTF2Infer:
    def __init__(self, cfg: TwoStageConfig):
        _enable_memory_growth()
        self.cfg = cfg

        # label maps (used for score keys)
        self.id2name_m2 = _parse_label_map_pbtxt(cfg.label_map_path_model2)
        self.id2name_m4 = _parse_label_map_pbtxt(cfg.label_map_path_model4)

        if not self.id2name_m2:
            raise ValueError(f"Could not parse label map: {cfg.label_map_path_model2}")
        if not self.id2name_m4:
            raise ValueError(f"Could not parse label map: {cfg.label_map_path_model4}")

        # detectors
        self.det_m1 = _TFODDetector(cfg.model1_savedmodel_dir)
        self.det_m2 = _TFODDetector(cfg.model2_savedmodel_dir)
        self.det_m3 = _TFODDetector(cfg.model3_savedmodel_dir)
        self.det_m4 = _TFODDetector(cfg.model4_savedmodel_dir)

        # temporal ROI carry state for video (Model-3 skip mode)
        self.prev_bbox_m3 = None   # full-frame normalized box [ymin,xmin,ymax,xmax]
        self.prev_ready_m3 = False
        self.prev_frame_shape = None

    def _load_no_roi(self) -> np.ndarray:
        """
        Loads noROI.jpg from current working directory.
        If not found, returns a simple gray placeholder.
        """
        p = os.path.join(os.getcwd(), "noROI.jpg")
        img = _imread_unicode(p)
        if img is None:
            # Fallback placeholder
            img = np.full((240, 320, 3), 200, dtype=np.uint8)
            cv2.putText(img, "noROI.jpg missing", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 200), 2)
        return img

    @staticmethod
    def _best_stage1(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray) -> Tuple[np.ndarray, float, int]:
        if scores.size == 0:
            return boxes[0], 0.0, int(classes[0]) if classes.size else 1
        i = int(np.argmax(scores))
        return boxes[i], float(scores[i]), int(classes[i])

    
    def _parts_union_bbox_full(self,
                               boxes_roi: np.ndarray,
                               scores: np.ndarray,
                               classes: np.ndarray,
                               roi_box_full: np.ndarray,
                               score_thresh: float,
                               min_parts: int = 3):
        """
        Build a bbox that covers the detected tag sub-parts (e.g., A/B/C/D) and project it to full-frame
        normalized coordinates.

        Parameters
        ----------
        boxes_roi : (N,4) normalized boxes in ROI image coordinates [ymin,xmin,ymax,xmax]
        scores/classes : detector outputs
        roi_box_full : (4,) normalized ROI crop box in full-frame coords [ymin,xmin,ymax,xmax]
        score_thresh : confidence threshold for parts
        min_parts : minimum number of parts required

        Returns
        -------
        np.ndarray (4,) full-frame normalized box, or None if not enough parts.
        """
        if boxes_roi is None or scores is None or classes is None:
            return None
        if len(boxes_roi) == 0 or len(scores) == 0 or len(classes) == 0:
            return None

        # Determine which part labels exist (prefer A/B/C/D if available)
        prefer = {"A", "B", "C", "D"}
        has_prefer = any(name in prefer for name in self.id2name_m4.values())
        ymin_r, xmin_r, ymax_r, xmax_r = [float(v) for v in roi_box_full.tolist()]
        rh = max(1e-9, ymax_r - ymin_r)
        rw = max(1e-9, xmax_r - xmin_r)

        kept = []
        for b, s, cid in zip(boxes_roi, scores, classes):
            if float(s) < float(score_thresh):
                continue
            name = self.id2name_m4.get(int(cid), None)
            if name is None:
                continue
            if has_prefer and (name not in prefer):
                continue
            kept.append(b)

        if len(kept) < int(min_parts):
            return None

        kept = np.asarray(kept, dtype=np.float32)
        y1 = float(np.min(kept[:, 0])); x1 = float(np.min(kept[:, 1]))
        y2 = float(np.max(kept[:, 2])); x2 = float(np.max(kept[:, 3]))

        # Project from ROI coordinates to full-frame coordinates.
        fy1 = ymin_r + y1 * rh
        fx1 = xmin_r + x1 * rw
        fy2 = ymin_r + y2 * rh
        fx2 = xmin_r + x2 * rw

        # Clamp to [0,1].
        fy1 = max(0.0, min(1.0, fy1))
        fx1 = max(0.0, min(1.0, fx1))
        fy2 = max(0.0, min(1.0, fy2))
        fx2 = max(0.0, min(1.0, fx2))

        # Validate the projected box.
        if fy2 <= fy1 or fx2 <= fx1:
            return None

        return np.array([fy1, fx1, fy2, fx2], dtype=np.float32)
    def infer(self, image_bgr: np.ndarray) -> InferResult:
        """
        Main inference:
            res = runner.infer(image)
            roi1 = res.roi1
            roi2 = res.roi2
            m1 = res.get("m1_score")
            m2 = res.get("m2_scores")
            ...
        """
        if image_bgr is None or not isinstance(image_bgr, np.ndarray) or image_bgr.ndim != 3:
            raise ValueError("infer(image) expects a BGR numpy image (HxWx3).")

        H, W = image_bgr.shape[:2]
        no_roi_img = self._load_no_roi()

        # ----------------- M1 -> ROI1 -> M2 -----------------
        b1, s1, c1 = self.det_m1.infer(image_bgr)
        box1, m1_score, _ = self._best_stage1(b1, s1, c1)

        if float(m1_score) < float(self.cfg.score_thresh_roi):
            roi1_vis = no_roi_img.copy()
            m2_scores = _collect_max_scores(
                scores=np.array([], dtype=np.float32),
                classes=np.array([], dtype=np.float32),
                id_to_name=self.id2name_m2,
                score_thresh=self.cfg.score_thresh_tag,
                fill_zero_for_all_labels=True
            )
        else:
            rx1, ry1, rx2, ry2 = _compute_roi_from_box(box1, W, H, self.cfg.roi_x)
            roi1 = image_bgr[ry1:ry2, rx1:rx2].copy()
            b2, s2, c2 = self.det_m2.infer(roi1)
            roi1_vis = roi1.copy()
            _draw_boxes_only(roi1_vis, b2, s2, c2, self.cfg.score_thresh_tag, thickness=None)
            m2_scores = _collect_max_scores(s2, c2, self.id2name_m2, self.cfg.score_thresh_tag, fill_zero_for_all_labels=True)

        # ----------------- M3 -> ROI2 -> M4 (with temporal ROI carry) -----------------
        # Video advantage: if previous frame produced a reliable parts-based bbox, reuse it to avoid M3 false-positives
        # when the tag is very small / far away.
        if getattr(self, "prev_frame_shape", None) != (H, W):
            # frame shape changed -> reset temporal state
            self.prev_bbox_m3 = None
            self.prev_ready_m3 = False
            self.prev_frame_shape = (H, W)

        use_prev = bool(getattr(self, "prev_ready_m3", False)) and (getattr(self, "prev_bbox_m3", None) is not None)

        def _run_m4_and_update(box3_norm: np.ndarray, roi_x_use: float) -> Tuple[np.ndarray, float, Dict[str, float], Optional[np.ndarray]]:
            """Run ROI2 crop -> resize -> M4. Returns (roi2_vis, roi2_scale, m4_scores, parts_bbox_full or None)."""
            nx1, ny1, nx2, ny2 = _compute_roi_from_box(box3_norm, W, H, roi_x_use)
            roi2 = image_bgr[ny1:ny2, nx1:nx2].copy()

            roi2_infer, roi2_scale_local = _resize_keep_aspect_long_side(roi2, self.cfg.roi2_long_side)
            b4, s4, c4 = self.det_m4.infer(roi2_infer)

            roi2_vis_local = roi2_infer.copy()
            _draw_boxes_only(roi2_vis_local, b4, s4, c4, self.cfg.score_thresh_tag, thickness=None)

            m4_scores_local = _collect_max_scores(
                s4, c4, self.id2name_m4, self.cfg.score_thresh_tag, fill_zero_for_all_labels=True
            )

            roi_box_full = np.array([ny1 / float(H), nx1 / float(W), ny2 / float(H), nx2 / float(W)], dtype=np.float32)
            parts_bbox_full = self._parts_union_bbox_full(
                boxes_roi=b4,
                scores=s4,
                classes=c4,
                roi_box_full=roi_box_full,
                score_thresh=float(self.cfg.score_thresh_tag),
                min_parts=3
            )
            return roi2_vis_local, float(roi2_scale_local), m4_scores_local, parts_bbox_full

        roi2_scale = 1.0

        if use_prev:
            # ---- 1) Temporal ROI attempt (skip M3) ----
            box3_prev = np.array(self.prev_bbox_m3, dtype=np.float32)
            
            # --- Size-aware temporal ROI expansion ---
            # NOTE: _compute_roi_from_box expands when roi_x is SMALLER (roi_w = bw / roi_x).
            # User-facing rule: if bbox area < 1000 px => expansion factor 2x (use roi_x / 2).
            # Otherwise, expand 60% more than base (use roi_x / 1.6).
            bx1, by1, bx2, by2 = [float(v) for v in box3_prev.tolist()]
            bw_px = max(1.0, (bx2 - bx1) * float(W))
            bh_px = max(1.0, (by2 - by1) * float(H))
            area_px = bw_px * bh_px

            if area_px < 1000.0:
                roi_x_use = max(1e-6, float(self.cfg.roi_x) / 2.0)
            else:
                roi_x_use = max(1e-6, float(self.cfg.roi_x) / 2.0)

            roi2_vis, roi2_scale, m4_scores, parts_bbox_full = _run_m4_and_update(box3_prev, roi_x_use)

            if parts_bbox_full is not None:
                # Temporal strategy succeeded: M3 was genuinely not used on this frame.
                m3_score = -1.0
                self.prev_bbox_m3 = parts_bbox_full
                self.prev_ready_m3 = True
            else:
                # Temporal strategy failed on THIS frame -> immediately fall back to M3 on the same frame.
                self.prev_bbox_m3 = None
                self.prev_ready_m3 = False

                b3, s3, c3 = self.det_m3.infer(image_bgr)
                box3, m3_score, _ = self._best_stage1(b3, s3, c3)

                if float(m3_score) < float(self.cfg.score_thresh_roi):
                    roi2_vis = no_roi_img.copy()
                    m4_scores = _collect_max_scores(
                        scores=np.array([], dtype=np.float32),
                        classes=np.array([], dtype=np.float32),
                        id_to_name=self.id2name_m4,
                        score_thresh=self.cfg.score_thresh_tag,
                        fill_zero_for_all_labels=True
                    )
                else:
                    roi2_vis, roi2_scale, m4_scores, parts_bbox_full2 = _run_m4_and_update(box3, float(self.cfg.roi_x))
                    if parts_bbox_full2 is not None:
                        self.prev_bbox_m3 = parts_bbox_full2
                        self.prev_ready_m3 = True
                    else:
                        self.prev_bbox_m3 = None
                        self.prev_ready_m3 = False

        else:
            # ---- 2) Normal path (use M3) ----
            b3, s3, c3 = self.det_m3.infer(image_bgr)
            box3, m3_score, _ = self._best_stage1(b3, s3, c3)

            if float(m3_score) < float(self.cfg.score_thresh_roi):
                roi2_vis = no_roi_img.copy()
                m4_scores = _collect_max_scores(
                    scores=np.array([], dtype=np.float32),
                    classes=np.array([], dtype=np.float32),
                    id_to_name=self.id2name_m4,
                    score_thresh=self.cfg.score_thresh_tag,
                    fill_zero_for_all_labels=True
                )
                self.prev_bbox_m3 = None
                self.prev_ready_m3 = False
            else:
                roi2_vis, roi2_scale, m4_scores, parts_bbox_full3 = _run_m4_and_update(box3, float(self.cfg.roi_x))
                if parts_bbox_full3 is not None:
                    self.prev_bbox_m3 = parts_bbox_full3
                    self.prev_ready_m3 = True
                else:
                    self.prev_bbox_m3 = None
                    self.prev_ready_m3 = False



        data = {
            "m1_score": float(m1_score),
            "m2_scores": m2_scores,
            "m3_score": float(m3_score),
            "m4_scores": m4_scores,
            "roi2_scale": float(roi2_scale),
            "roi_x": float(self.cfg.roi_x),
            "roi2_long_side": int(self.cfg.roi2_long_side),
        }
        return InferResult(data=data, roi1=roi1_vis, roi2=roi2_vis)
