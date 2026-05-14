
# -*- coding: utf-8 -*-
"""
app_video_compare.py

PyQt5 interface for comparing video-frame inference results from the two-stage
detection pipeline.

The interface shows:
- Left panel: original frame with two semi-transparent score panels
- Right panel: ROI views for the previous and proposed models

When recording is enabled, the full PyQt window is captured with QWidget.grab()
and saved as an MP4 video.

Dependencies:
- PyQt5, opencv-python, numpy
- two_stage_infer_class.py (TwoStageTF2Infer, TwoStageConfig)
- score_panel_overlay.py (ScorePanelOverlay)
"""

import os
import sys
import traceback
import csv
import time

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QHBoxLayout, QVBoxLayout, QGroupBox, QSlider, QMessageBox
)



from two_stage_infer_class import TwoStageTF2Infer, TwoStageConfig
from score_panel_overlay import ScorePanelOverlay


# -----------------------------
# Fixed settings
# -----------------------------
FRAME_STRIDE = 1            # process every Nth frame (increase if slow)
ROI_VIEW_W = 250
ROI_PANEL_W = 320  # fixed display box width for ROI panels
ROI_PANEL_H = 240  # fixed display box height for ROI panels            # ROI1/ROI2 display width (keep aspect)
LEFT_MAX_W = 980            # max display width for out_img
TIMER_MS = 1                # timer tick; actual FPS limited by inference time


def resize_keep_aspect_by_width(img_bgr, target_w=250):
    if img_bgr is None:
        return None
    h, w = img_bgr.shape[:2]
    if w <= 0:
        return img_bgr
    scale = target_w / float(w)
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img_bgr, (target_w, new_h), interpolation=cv2.INTER_LINEAR)


def bgr_to_qimage(img_bgr):
    if img_bgr is None:
        return QImage()
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    bytes_per_line = ch * w
    return QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()


def qimage_to_bgr(qimg):
    """
    Convert QImage -> BGR uint8 (OpenCV).
    Uses RGBA8888 to guarantee 4 channels.
    """
    if qimg is None or qimg.isNull():
        return None

    qimg = qimg.convertToFormat(QImage.Format_RGBA8888)
    w = qimg.width()
    h = qimg.height()

    ptr = qimg.bits()
    # PyQt needs explicit buffer size
    ptr.setsize(h * w * 4)
    arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))  # RGBA
    rgb = arr[:, :, :3]
    bgr = rgb[:, :, ::-1].copy()
    return bgr



def load_label_map_pbtxt(path):
    """
    Minimal TFOD label_map.pbtxt parser.
    Returns:
        id_to_name: dict[int, str]
        names_sorted: list[str] in increasing id order
    """
    id_to_name = {}
    if not path or not os.path.exists(path):
        return id_to_name, []
    cur_id = None
    cur_name = None
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s.startswith("id:"):
                try:
                    cur_id = int(s.split(":", 1)[1].strip())
                except Exception:
                    cur_id = None
            elif s.startswith("name:") or s.startswith("display_name:"):
                # name: "A"
                try:
                    v = s.split(":", 1)[1].strip()
                    v = v.strip('"').strip("'")
                    cur_name = v
                except Exception:
                    cur_name = None
            elif s.startswith("}"):
                if cur_id is not None and cur_name:
                    id_to_name[cur_id] = cur_name
                cur_id = None
                cur_name = None
    names_sorted = [id_to_name[k] for k in sorted(id_to_name.keys())]
    return id_to_name, names_sorted


class ImageLabel(QLabel):
    """A QLabel that shows BGR images via QPixmap (keeps aspect) inside a fixed box."""
    def __init__(self, fixed_w=None, fixed_h=None, min_h=240):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background:#111; border:1px solid #444;")
        self.setScaledContents(False)

        if fixed_w is not None and fixed_h is not None:
            self.setFixedSize(int(fixed_w), int(fixed_h))
        else:
            self.setMinimumHeight(min_h)

    def set_image_bgr(self, img_bgr, max_w=None):
        if img_bgr is None:
            self.clear()
            return
        qimg = bgr_to_qimage(img_bgr)
        pix = QPixmap.fromImage(qimg)

        # If this label has a fixed size, always scale pixmap to fit inside it
        if self.maximumWidth() == self.minimumWidth() and self.maximumHeight() == self.minimumHeight():
            target_w = self.width()
            target_h = self.height()
            pix = pix.scaled(target_w, target_h, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            if max_w is not None and pix.width() > max_w:
                pix = pix.scaledToWidth(max_w, Qt.SmoothTransformation)

        self.setPixmap(pix)


class VideoCompareApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Frame Compare (Full GUI Recording)")
        self.resize(1400, 720)

        # ---- Model configuration ----
        cfg_fields = set(getattr(TwoStageConfig, "__dataclass_fields__", {}).keys())

        kwargs = dict(
            model1_savedmodel_dir=r"Models\M1",  # previous ROI model
            model2_savedmodel_dir=r"Models\M2",  # previous sub-component model
            model3_savedmodel_dir=r"Models\M5",  # proposed ROI model
            model4_savedmodel_dir=r"Models\M4",  # proposed sub-component model

            label_map_path_model1=r"Models\M1\label_map.pbtxt",
            label_map_path_model2=r"Models\M2\label_map.pbtxt",
            label_map_path_model3=r"Models\M5\label_map.pbtxt",
            label_map_path_model4=r"Models\M4\label_map.pbtxt",
            roi_x=0.60,
            roi2_long_side=512,
        )

        # separated threshold configuration
        if "score_thresh_roi" in cfg_fields and "score_thresh_tag" in cfg_fields:
            kwargs["score_thresh_roi"] = 0.50
            kwargs["score_thresh_tag"] = 0.45
        # shared threshold configuration
        elif "score_thresh" in cfg_fields:
            kwargs["score_thresh"] = 0.45

        cfg = TwoStageConfig(**kwargs)
        self.runner = TwoStageTF2Infer(cfg)

        self.overlay = ScorePanelOverlay(
            label_map_parts_old=r"Models\M2\label_map.pbtxt",
            label_map_parts_new=r"Models\M4\label_map.pbtxt",
            panel_w=360,
            panel_h=260,
            margin=18,
            gap_between_panels=18
        )

        # ---- CSV logging ----
        # Save per-frame scores to CSV (frame index -> scores)
        self.csv_path = None
        self.csv_fp = None
        self.csv_writer = None
        self.csv_logged_frames = set()
        self.id2name_old, self.names_old = load_label_map_pbtxt(r"Models\M2\label_map.pbtxt")
        self.id2name_new, self.names_new = load_label_map_pbtxt(r"Models\M4\label_map.pbtxt")

        # ---- Video state ----
        self.cap = None
        self.video_path = None
        self.total_frames = 0
        self.fps = 0.0
        self.cur_frame_idx = 0
        self.playing = False
        self._block_slider = False
        self._stride_counter = 0

        # ---- Recording state (FULL GUI) ----
        self.recording = False
        self.writer = None
        self._pending_writer_path = None
        self._locked_size = None  # (w,h) while recording

        # ---- UI ----
        self.btn_open = QPushButton("Video")
        self.btn_play = QPushButton("Play")
        self.btn_step = QPushButton("Step +1")
        self.btn_rec = QPushButton("REC OFF")
        self.lbl_info = QLabel("No video loaded.")
        self.lbl_info.setStyleSheet("color:#333;")

        self.btn_open.clicked.connect(self.on_open_video)
        self.btn_play.clicked.connect(self.on_toggle_play)
        self.btn_step.clicked.connect(self.on_step)
        self.btn_rec.clicked.connect(self.on_toggle_record)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        self.slider.valueChanged.connect(self.on_slider_changed)

        top = QWidget()
        top_l = QHBoxLayout(top)
        top_l.setContentsMargins(8, 8, 8, 4)
        top_l.setSpacing(8)
        top_l.addWidget(self.btn_open)
        top_l.addWidget(self.btn_play)
        top_l.addWidget(self.btn_step)
        top_l.addWidget(self.btn_rec)
        top_l.addWidget(self.lbl_info, 1)

        # Main image view
        self.lbl_out = ImageLabel(min_h=520)

        # ROI views
        self.lbl_roi1 = ImageLabel(fixed_w=ROI_PANEL_W, fixed_h=ROI_PANEL_H)
        self.lbl_roi2 = ImageLabel(fixed_w=ROI_PANEL_W, fixed_h=ROI_PANEL_H)

        box1 = QGroupBox("Previous Model (Real Dataset) — ROI view")
        box1.setFixedWidth(ROI_PANEL_W + 30)

        b1 = QVBoxLayout(box1)
        b1.setContentsMargins(8, 12, 8, 8)
        b1.addWidget(self.lbl_roi1)

        box2 = QGroupBox("New Model (Synthetic Dataset) — ROI view")
        box2.setFixedWidth(ROI_PANEL_W + 30)

        b2 = QVBoxLayout(box2)
        b2.setContentsMargins(8, 12, 8, 8)
        b2.addWidget(self.lbl_roi2)

        right = QWidget()
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(8, 0, 8, 8)
        right_l.setSpacing(10)
        right_l.addWidget(box1)
        right_l.addWidget(box2)
        right_l.addStretch(1)

        center = QWidget()
        center_l = QHBoxLayout(center)
        center_l.setContentsMargins(8, 4, 8, 4)
        center_l.setSpacing(10)
        center_l.addWidget(self.lbl_out, 3)
        center_l.addWidget(right, 1)

        root = QWidget()
        root_l = QVBoxLayout(root)
        root_l.setContentsMargins(0, 0, 0, 0)
        root_l.setSpacing(0)
        root_l.addWidget(top)
        root_l.addWidget(self.slider)
        root_l.addWidget(center, 1)

        self.setCentralWidget(root)

        # ---- Timer loop ----
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.on_tick)
        self.timer.start(TIMER_MS)

    # -----------------------------
    # Recording (full GUI)
    # -----------------------------
    def on_toggle_record(self):
        if self.cap is None or self.video_path is None:
            QMessageBox.information(self, "Info", "Select a video first.")
            return

        if not self.recording:
            base = os.path.splitext(os.path.basename(self.video_path))[0]
            out_path = os.path.join(os.getcwd(), f"{base}_FULLGUI.mp4")

            # lock window size to keep constant resolution in output video
            self._locked_size = (self.width(), self.height())
            self.setFixedSize(self.width(), self.height())

            self.recording = True
            self.btn_rec.setText("REC ON")
            self._pending_writer_path = out_path
            if self.writer is not None:
                self.writer.release()
            self.writer = None  # open on first grabbed frame

        else:
            self.recording = False
            self.btn_rec.setText("REC OFF")

            # unlock resizing
            self.setMinimumSize(0, 0)
            self.setMaximumSize(16777215, 16777215)  # Qt default max
            if self._locked_size:
                # keep current size (do not force resize), but release the lock
                self._locked_size = None

            if self.writer is not None:
                self.writer.release()
                self.writer = None

            QMessageBox.information(self, "Saved", "Full GUI video recording has been stopped.")

    def _grab_full_gui_bgr(self):
        """
        Capture the entire QMainWindow (buttons, slider, images) as BGR.
        Note: Window must be visible; minimized windows may capture blank frames.
        """
        pix = self.grab()
        # handle HiDPI
        dpr = pix.devicePixelRatio()
        qimg = pix.toImage()
        if dpr and dpr != 1.0:
            qimg = qimg.scaled(int(qimg.width() * dpr), int(qimg.height() * dpr),
                               Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        return qimage_to_bgr(qimg)

    def _write_gui_frame(self, gui_frame_bgr):
        if not self.recording or gui_frame_bgr is None:
            return

        if self.writer is None:
            h, w = gui_frame_bgr.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps = self.fps if self.fps and self.fps > 0 else 25.0
            self.writer = cv2.VideoWriter(self._pending_writer_path, fourcc, fps, (w, h))

        self.writer.write(gui_frame_bgr)


    # -----------------------------
    # CSV logging helpers
    # -----------------------------
    def _open_csv_for_video(self, video_path):
        """Open CSV in current working dir with a name based on video."""
        self._close_csv()
        base = os.path.splitext(os.path.basename(video_path))[0]
        self.csv_path = os.path.join(os.getcwd(), f"{base}.csv")
        self.csv_fp = open(self.csv_path, "w", newline="", encoding="utf-8")
        self.csv_writer = csv.DictWriter(self.csv_fp, fieldnames=self._csv_header())
        self.csv_writer.writeheader()
        self.csv_fp.flush()
        self.csv_logged_frames = set()

    def _close_csv(self):
        try:
            if self.csv_fp is not None:
                self.csv_fp.flush()
                self.csv_fp.close()
        except Exception:
            pass
        self.csv_fp = None
        self.csv_writer = None

    def _csv_header(self):
        # frame metadata + ROI scores + part scores
        cols = [
            "frame_idx",
            "time_sec",
            "m1_score",
            "m3_score",
        ]
        # Old parts (M2)
        for name in self.names_old:
            cols.append(f"old_{name}")
        # New parts (M4)
        for name in self.names_new:
            cols.append(f"new_{name}")
        return cols

    def _log_scores_to_csv(self, frame_idx, time_sec, m1, m2_dict, m3, m4_dict):
        if self.csv_writer is None:
            return
        # Avoid duplicate rows during slider navigation.
        if frame_idx in self.csv_logged_frames:
            return
        self.csv_logged_frames.add(frame_idx)

        row = {
            "frame_idx": int(frame_idx),
            "time_sec": float(time_sec),
            "m1_score": float(m1) if m1 is not None else 0.0,
            "m3_score": float(m3) if m3 is not None else 0.0,
        }

        # Initialize previous/proposed scores with zeros.
        for name in self.names_old:
            row[f"old_{name}"] = 0.0
        for name in self.names_new:
            row[f"new_{name}"] = 0.0

        # m2_dict/m4_dict may use ids or names as keys
        if isinstance(m2_dict, dict):
            for k, v in m2_dict.items():
                try:
                    sc = float(v) if v is not None else 0.0
                except Exception:
                    sc = 0.0
                if isinstance(k, str):
                    row_key = f"old_{k}"
                    if row_key in row:
                        row[row_key] = sc
                else:
                    try:
                        kid = int(k)
                        name = self.id2name_old.get(kid)
                        if name is not None:
                            row[f"old_{name}"] = sc
                    except Exception:
                        pass

        if isinstance(m4_dict, dict):
            for k, v in m4_dict.items():
                try:
                    sc = float(v) if v is not None else 0.0
                except Exception:
                    sc = 0.0
                if isinstance(k, str):
                    row_key = f"new_{k}"
                    if row_key in row:
                        row[row_key] = sc
                else:
                    try:
                        kid = int(k)
                        name = self.id2name_new.get(kid)
                        if name is not None:
                            row[f"new_{name}"] = sc
                    except Exception:
                        pass

        self.csv_writer.writerow(row)
        # flush periodically
        if frame_idx % 10 == 0:
            try:
                self.csv_fp.flush()
            except Exception:
                pass

    # -----------------------------
    # Video helpers
    # -----------------------------
    def on_open_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select video", "", "Video (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        self.load_video(path)

    def load_video(self, path):
        try:
            if self.cap is not None:
                self.cap.release()
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                raise RuntimeError("Could not open video.")

            self.video_path = path
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.fps = float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)

            # open CSV log for this video
            self._open_csv_for_video(path)

            self.cur_frame_idx = 0
            self._stride_counter = 0
            self.playing = False
            self.btn_play.setText("Play")

            self._block_slider = True
            self.slider.setMaximum(max(0, self.total_frames - 1))
            self.slider.setValue(0)
            self._block_slider = False

            self.lbl_info.setText(
                f"{os.path.basename(path)} | Frames: {self.total_frames} | FPS: {self.fps:.2f} | Stride: {FRAME_STRIDE}"
            )
            self.show_frame_at(0)
        except Exception as e:
            QMessageBox.critical(self, "Video Load Error", f"{e}\n\n{traceback.format_exc()}")

    def on_toggle_play(self):
        if self.cap is None:
            return
        self.playing = not self.playing
        self.btn_play.setText("Pause" if self.playing else "Play")

    def on_step(self):
        if self.cap is None:
            return
        self.playing = False
        self.btn_play.setText("Play")
        self.show_frame_at(min(self.total_frames - 1, self.cur_frame_idx + 1))

    def on_slider_changed(self, val):
        if self._block_slider or self.cap is None:
            return
        self.playing = False
        self.btn_play.setText("Play")
        self.show_frame_at(int(val))

    def on_tick(self):
        if not self.playing or self.cap is None:
            return

        self._stride_counter = (self._stride_counter + 1) % FRAME_STRIDE
        nxt = self.cur_frame_idx + 1
        if nxt >= self.total_frames:
            self.playing = False
            self.btn_play.setText("Play")
            return

        if self._stride_counter != 0:
            # advance without heavy processing
            self.cur_frame_idx = nxt
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.cur_frame_idx)
            self._sync_slider()
            return

        self.show_frame_at(nxt)

    def _sync_slider(self):
        self._block_slider = True
        self.slider.setValue(self.cur_frame_idx)
        self._block_slider = False

    def show_frame_at(self, idx):
        idx = max(0, min(idx, self.total_frames - 1))
        self.cur_frame_idx = idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return

        try:
            res = self.runner.infer(frame)

            # Support both return types:
            # - dict: {"roi1":..., "roi2":..., "m1_score":..., ...}
            # - object with attributes: res.roi1, res.roi2 and/or res.get(...)
            if isinstance(res, dict):
                # Different versions may use different key names
                roi1_raw = (res.get("roi1") or res.get("roi1_bgr") or res.get("roiImage1") or
                            res.get("roi_img1") or res.get("roi_stage1") or res.get("roi"))
                roi2_raw = (res.get("roi2") or res.get("roi2_bgr") or res.get("roiImage2") or
                            res.get("roi_img2") or res.get("roi_stage2"))

                # Scores (support multiple key variants)
                m1 = res.get("m1_score")
                if m1 is None:
                    m1 = res.get("model1_score")

                m2 = res.get("m2_scores")
                if m2 is None:
                    m2 = res.get("model2_scores")

                m3 = res.get("m3_score")
                if m3 is None:
                    m3 = res.get("model3_score")

                m4 = res.get("m4_scores")
                if m4 is None:
                    # Some variants use a different key for proposed sub-component scores.
                    m4 = res.get("m4_scores") or res.get("m3_scores") or res.get("model4_scores")
            else:
                roi1_raw = getattr(res, "roi1", None)
                roi2_raw = getattr(res, "roi2", None)
                get_fn = getattr(res, "get", None)
                if callable(get_fn):
                    m1 = get_fn("m1_score")
                    m2 = get_fn("m2_scores")
                    m3 = get_fn("m3_score")
                    m4 = get_fn("m4_scores")
                else:
                    m1 = getattr(res, "m1_score", None)
                    m2 = getattr(res, "m2_scores", None)
                    m3 = getattr(res, "m3_score", None)
                    m4 = getattr(res, "m4_scores", None)

            roi1 = resize_keep_aspect_by_width(roi1_raw, ROI_VIEW_W)
            roi2 = resize_keep_aspect_by_width(roi2_raw, ROI_VIEW_W)

            # Sanitize scores (overlay expects numbers; sometimes None comes from missing detections)
            def _safe_float(v, default=0.0):
                try:
                    if v is None:
                        return float(default)
                    return float(v)
                except Exception:
                    return float(default)

            m1 = _safe_float(m1, 0.0)
            m3 = _safe_float(m3, 0.0)

            if m2 is None:
                m2 = {}
            if m4 is None:
                m4 = {}

            # Convert score dict values to floats (None -> 0.0)
            if isinstance(m2, dict):
                m2 = {k: _safe_float(v, 0.0) for k, v in m2.items()}
            if isinstance(m4, dict):
                m4 = {k: _safe_float(v, 0.0) for k, v in m4.items()}


            # Log frame-level scores to CSV.
            time_sec = float(self.cur_frame_idx) / float(self.fps if self.fps and self.fps > 0 else 1.0)
            self._log_scores_to_csv(self.cur_frame_idx, time_sec, m1, m2, m3, m4)

            out_img = self.overlay.render(
                image_bgr=frame,
                m1_score=m1,
                m2_scores=m2,
                m3_score=m3,
                m4_scores=m4,
                title_prev="Previous Model (Real Dataset)",
                title_new="New Model (Synthetic Dataset)"
            )

            # Update UI
            self.lbl_out.set_image_bgr(out_img, max_w=LEFT_MAX_W)
            self.lbl_roi1.set_image_bgr(roi1)
            self.lbl_roi2.set_image_bgr(roi2)

            # Record full GUI after UI update
            if self.recording:
                gui_bgr = self._grab_full_gui_bgr()
                self._write_gui_frame(gui_bgr)

        except Exception as e:
            self.playing = False
            self.btn_play.setText("Play")
            QMessageBox.critical(self, "Inference Error", f"{e}\n\n{traceback.format_exc()}")

        self._sync_slider()

    def closeEvent(self, event):
        # Release video and recording resources.
        try:
            if self.writer is not None:
                self.writer.release()
                self.writer = None
        except Exception:
            pass
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
        except Exception:
            pass
        event.accept()


def main():
    app = QApplication(sys.argv)
    w = VideoCompareApp()
    w.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
