
# -*- coding: utf-8 -*-
"""
score_panel_overlay.py

Creates two semi-transparent score panels on the original image:
- Top-left: Previous Model (Real Dataset)
- Bottom-left: New Model (Synthetic Dataset)

This class does not run inference. It only overlays score panels using the
inference results. Text colors follow the same class-id-based HSV rule used for
bounding-box colors.

Input: original BGR image and inference score dictionaries
Output: BGR image with two semi-transparent score panels
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import cv2
import numpy as np


# ---------------------------
# Label map parsing (id <-> name)
# ---------------------------

_ITEM_RE = re.compile(r"item\s*\{.*?\}", flags=re.S)
_ID_RE = re.compile(r"\bid\s*:\s*(\d+)")
_NAME_RE = re.compile(r"\bname\s*:\s*['\"]([^'\"]+)['\"]")
_DNAME_RE = re.compile(r"\bdisplay_name\s*:\s*['\"]([^'\"]+)['\"]")


def parse_label_map(pbtxt_path: str) -> Tuple[Dict[int, str], Dict[str, int]]:
    if not pbtxt_path or not os.path.exists(pbtxt_path):
        return {}, {}
    txt = open(pbtxt_path, "r", encoding="utf-8", errors="ignore").read()
    items = _ITEM_RE.findall(txt)
    id2name: Dict[int, str] = {}
    name2id: Dict[str, int] = {}
    for it in items:
        mid = _ID_RE.search(it)
        if not mid:
            continue
        cid = int(mid.group(1))
        d = _DNAME_RE.search(it)
        n = _NAME_RE.search(it)
        name = (d.group(1).strip() if d else (n.group(1).strip() if n else str(cid)))
        id2name[cid] = name
        name2id.setdefault(name, cid)
    return id2name, name2id


# ---------------------------
# BBox color rule
# ---------------------------

def hsv_color_from_id(class_id: int) -> Tuple[int, int, int]:
    """Return a deterministic BGR color from the class ID."""
    h = int((class_id * 37) % 180)
    s = 220
    v = 255
    hsv = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


# ---------------------------
# Overlay class
# ---------------------------

# @dataclass
# class PanelStyle:
#     bg_bgr: Tuple[int, int, int] = (0, 0, 0)  # black base
#     alpha: float = 0.55                       # transparency
#     pad: int = 14
#     font: int = cv2.FONT_HERSHEY_SIMPLEX
#     title_scale: float = 0.7
#     title_th: int = 2
#     text_scale: float = 0.75
#     text_th: int = 2
    
@dataclass
class PanelStyle:
    bg_bgr: Tuple[int, int, int] = (0, 0, 0)
    alpha: float = 0.55
    pad: int = 14
    font: int = cv2.FONT_HERSHEY_DUPLEX
    title_scale: float = 0.65
    title_th: int = 1
    text_scale: float = 0.70
    text_th: int = 1
    
    


class ScorePanelOverlay:
    def __init__(self,
                 label_map_parts_old: str,
                 label_map_parts_new: str,
                 panel_w: int = 390,
                 panel_h: int = 290,
                 margin: int = 18,
                 gap_between_panels: int = 18,
                 style: Optional[PanelStyle] = None):
        self.panel_w = int(panel_w)
        self.panel_h = int(panel_h)
        self.margin = int(margin)
        self.gap = int(gap_between_panels)
        self.style = style or PanelStyle()

        _, self.name2id_old = parse_label_map(label_map_parts_old)
        _, self.name2id_new = parse_label_map(label_map_parts_new)

    def _color_for_label(self, label: str, which: str) -> Tuple[int, int, int]:
        cid = (self.name2id_old.get(label) if which == "old" else self.name2id_new.get(label))
        if cid is None:
            return (255, 255, 255)
        return hsv_color_from_id(int(cid))

    def _put_text(self, img: np.ndarray, text: str, org: Tuple[int, int],
                  color: Tuple[int, int, int], scale: float, th: int) -> None:
        cv2.putText(img, text, org, self.style.font, scale, color, th, cv2.LINE_AA)

    def _draw_panel(self,
                    canvas: np.ndarray,
                    top_left: Tuple[int, int],
                    title: str,
                    abcd_score: Optional[float],
                    part_scores: Dict[str, float],
                    which: str):
        x0, y0 = top_left
        x1 = min(canvas.shape[1], x0 + self.panel_w)
        y1 = min(canvas.shape[0], y0 + self.panel_h)

        # semi-transparent rectangle
        overlay = canvas.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), self.style.bg_bgr, thickness=-1)
        cv2.addWeighted(overlay, self.style.alpha, canvas, 1.0 - self.style.alpha, 0, canvas)

        # text
        x = x0 + self.style.pad
        y = y0 + self.style.pad + 20

        self._put_text(canvas, title, (x, y), (255, 255, 255), self.style.title_scale, self.style.title_th)
        y += 44

        abcd_txt = "ABCD Score: -" if abcd_score is None else f"ABCD Score: {abcd_score:.2f}"
        abcd_col = (255, 255, 255)
        # Special marker: -1.00 means ABCD score is not from the ROI detector (M3 skipped via temporal ROI carry)
        if abcd_score is not None and float(abcd_score) < 0.0:
            abcd_col = (0, 255, 0)
        self._put_text(canvas, abcd_txt, (x, y), abcd_col, self.style.text_scale, self.style.text_th)
        y += 54

        # order: A,B,C,D if available, else sorted
        keys = list(part_scores.keys()) if part_scores else []
        prefer = ["A", "B", "C", "D"]
        order = prefer if keys and all(k in part_scores for k in prefer) else sorted(keys)

        for k in order:
            v = float(part_scores.get(k, 0.0))
            col = self._color_for_label(k, which)
            self._put_text(canvas, f"{k} Score: {v:.2f}", (x, y), col, self.style.text_scale, self.style.text_th)
            y += 34

    def render(self,
               image_bgr: np.ndarray,
               m1_score: Optional[float],
               m2_scores: Dict[str, float],
               m3_score: Optional[float],
               m4_scores: Dict[str, float],
               title_prev: str = "Previous Model (Real Dataset)",
               title_new: str = "New Model (Synthetic Dataset)") -> np.ndarray:
        if image_bgr is None:
            raise ValueError("image_bgr is None")
        out = image_bgr.copy()

        top1 = (self.margin, self.margin)
        top2 = (self.margin, self.margin + self.panel_h + self.gap)

        self._draw_panel(out, top1, title_prev, m1_score, m2_scores, which="old")
        self._draw_panel(out, top2, title_new, m3_score, m4_scores, which="new")

        return out
