# tag_embed_module.py
# -*- coding: utf-8 -*-

import cv2
import numpy as np

from background_profile import compute_patch_profile


# -----------------------------
# 3D Warp + Mask
# -----------------------------
def warp_and_crop_with_mask(img,
                            yaw_deg=0.0,
                            pitch_deg=0.0,
                            roll_deg=0.0,
                            fov_deg=60.0,
                            dist_scale=3.0,
                            border_color=(200, 200, 200)):
    h, w = img.shape[:2]

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

    H = cv2.getPerspectiveTransform(src, dst)

    warped_img = cv2.warpPerspective(
        img, H, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_color
    )

    src_mask = np.ones((h, w), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(
        src_mask, H, (w, h),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    mask_bin = warped_mask > 0
    if not np.any(mask_bin):
        return warped_img, (warped_mask > 0).astype(np.uint8)

    ys, xs = np.where(mask_bin)
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()

    tag_crop  = warped_img[y_min:y_max+1, x_min:x_max+1]
    mask_crop = warped_mask[y_min:y_max+1, x_min:x_max+1]
    mask_crop = (mask_crop > 0).astype(np.uint8)

    return tag_crop, mask_crop


# -----------------------------
# ROI-Profile Aware Tag Adapt
# -----------------------------
def _match_luminance_contrast(tag_bgr: np.ndarray, roi_prof, strength: float) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))

    lab = cv2.cvtColor(tag_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = cv2.split(lab)

    t_mean = float(np.mean(L))
    t_std  = float(np.std(L)) + 1e-6

    target_mean = roi_prof.mean_L
    target_std  = max(roi_prof.std_L, 6.0)

    L_adj = (L - t_mean) * (target_std / t_std) + target_mean
    L_out = (1.0 - strength) * L + strength * L_adj

    lab2 = cv2.merge([L_out, a, b])
    return cv2.cvtColor(np.clip(lab2, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _match_color_ab(tag_bgr: np.ndarray, roi_prof, strength: float) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))

    lab = cv2.cvtColor(tag_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, a, b = cv2.split(lab)

    a_mean = float(np.mean(a))
    b_mean = float(np.mean(b))

    a_adj = a + (roi_prof.mean_a - a_mean)
    b_adj = b + (roi_prof.mean_b - b_mean)

    a_out = (1.0 - strength) * a + strength * a_adj
    b_out = (1.0 - strength) * b + strength * b_adj

    lab2 = cv2.merge([L, a_out, b_out])
    return cv2.cvtColor(np.clip(lab2, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _match_sharpness(tag_bgr: np.ndarray, roi_prof) -> np.ndarray:
    if roi_prof.lap_var < 150.0:
        return cv2.GaussianBlur(tag_bgr, (3, 3), 0)
    return tag_bgr


def _add_noise_like_roi(tag_bgr: np.ndarray, roi_prof, strength: float, rng: np.random.Generator) -> np.ndarray:
    strength = float(np.clip(strength, 0.0, 1.0))

    target = float(roi_prof.noise)
    if target < 6.0:
        return tag_bgr

    noise_std = float(np.clip(target * 1.1, 3.0, 18.0))
    noise_kind = int(rng.integers(0, 3))  # 0 gaussian, 1 laplace, 2 speckle
    img = tag_bgr.astype(np.float32)

    if noise_kind == 0:
        n = rng.normal(0.0, noise_std, img.shape).astype(np.float32)
        out = img + strength * n
    elif noise_kind == 1:
        n = rng.laplace(0.0, noise_std / 1.4, img.shape).astype(np.float32)
        out = img + strength * n
    else:
        n = rng.normal(0.0, noise_std / 255.0, img.shape).astype(np.float32)
        out = img + strength * (img * n)

    return np.clip(out, 0, 255).astype(np.uint8)


def adapt_tag_to_scene_roi(tag_crop_bgr: np.ndarray,
                           scene_bgr: np.ndarray,
                           roi_box,
                           rng: np.random.Generator) -> np.ndarray:
    roi_prof = compute_patch_profile(scene_bgr, roi=roi_box)

    lum_strength   = float(rng.uniform(0.60, 0.95))
    color_strength = float(rng.uniform(0.30, 0.80))
    noise_strength = float(rng.uniform(0.25, 0.85))

    out = tag_crop_bgr
    out = _match_luminance_contrast(out, roi_prof, strength=lum_strength)
    out = _match_color_ab(out, roi_prof, strength=color_strength)
    out = _match_sharpness(out, roi_prof)
    out = _add_noise_like_roi(out, roi_prof, strength=noise_strength, rng=rng)
    return out


# -----------------------------
# Alpha / edge blending variants and parameter debug
# -----------------------------
def _alpha_gaussian(mask_u8: np.ndarray, rng: np.random.Generator):
    feather = int(rng.integers(3, 11))
    if feather % 2 == 0:
        feather += 1
    m = cv2.GaussianBlur(mask_u8, (feather, feather), 0)
    info = f"k={feather}"
    return (m.astype(np.float32) / 255.0)[..., None], info


# def _alpha_distance_feather(mask_u8: np.ndarray, rng: np.random.Generator):
#     """
#     Distance-based method keeps the edge from disappearing too much:
#     - alpha remains close to 1 inside core_px,
#     - and decreases only in the feather band after the core.
#     """
#     feather_px = float(rng.uniform(2.0, 6.0))
#     core_px = float(rng.uniform(1.0, 3.5))
#     gamma = float(rng.uniform(0.9, 1.2))

#     m = (mask_u8 > 0).astype(np.uint8)
#     dist_in = cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float32)

#     t = (dist_in - core_px) / max(feather_px, 1e-6)
#     alpha = np.clip(t, 0.0, 1.0) ** gamma

#     info = f"c={core_px:.1f},f={feather_px:.1f},g={gamma:.2f}"
#     return alpha[..., None], info

def _alpha_distance_feather(mask_u8: np.ndarray, rng: np.random.Generator):
    """
    Keep the distance-based alpha method stable for small tags.

    The feather width is capped using dmax to reduce edge erosion.
    The core size is floored using dmax to avoid an overly small core.
    """
    m = (mask_u8 > 0).astype(np.uint8)
    dist_in = cv2.distanceTransform(m, cv2.DIST_L2, 5).astype(np.float32)

    dmax = float(dist_in.max() if dist_in.size else 0.0)
    dmax = max(dmax, 1.0)

    # Base sampling
    feather_px = float(rng.uniform(2.0, 6.0))
    core_px    = float(rng.uniform(1.0, 3.5))
    gamma      = float(rng.uniform(0.9, 1.2))

    # Scale-based limits for small objects
    feather_cap = 0.22 * dmax          # Example: dmax=18.6 gives cap about 4.1
    core_floor  = 0.10 * dmax          # Example: dmax=18.6 gives floor about 1.86

    feather_px = min(feather_px, feather_cap)
    core_px    = max(core_px, core_floor)

    # Extra safety: keep core + feather below dmax
    limit = 0.92 * dmax
    if core_px + feather_px > limit:
        feather_px = max(0.6, limit - core_px)

    t = (dist_in - core_px) / max(feather_px, 1e-6)
    alpha = np.clip(t, 0.0, 1.0) ** gamma

    info = f"dmax={dmax:.1f},c={core_px:.2f},f={feather_px:.2f},g={gamma:.2f}"
    return alpha[..., None], info



def _alpha_morphological(mask_u8: np.ndarray, rng: np.random.Generator):
    m = (mask_u8 > 0).astype(np.uint8) * 255
    k = int(rng.integers(2, 6))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*k+1, 2*k+1))

    if rng.random() < 0.5:
        m2 = cv2.erode(m, kernel, iterations=1)
        op = "er"
    else:
        m2 = cv2.dilate(m, kernel, iterations=1)
        op = "di"

    blur_k = int(rng.integers(3, 9))
    if blur_k % 2 == 0:
        blur_k += 1
    m3 = cv2.GaussianBlur(m2, (blur_k, blur_k), 0)

    info = f"op={op},k={k},b={blur_k}"
    return (m3.astype(np.float32) / 255.0)[..., None], info


# def build_alpha(mask01: np.ndarray, rng: np.random.Generator):
#     """
#     Returns:
#         alpha: (h,w,1)
#         method: "gauss" / "dist" / "morph"
#         params: short debug string
#     """
#     mask_u8 = (mask01.astype(np.uint8) * 255)

#     method = int(rng.integers(0, 3))
#     if method == 0:
#         alpha, p = _alpha_gaussian(mask_u8, rng)
#         return alpha, "gauss", p
#     elif method == 1:
#         alpha, p = _alpha_distance_feather(mask_u8, rng)
#         return alpha, "dist", p
#     else:
#         alpha, p = _alpha_morphological(mask_u8, rng)
#         return alpha, "morph", p

def build_alpha(mask01: np.ndarray, rng: np.random.Generator, force_method: str = "random"):
    """
    force_method:
      - "random"  -> current random selection
      - "gauss" / "dist" / "morph" -> fixed single method
    """
    mask_u8 = (mask01.astype(np.uint8) * 255)

    methods = ["gauss", "dist", "morph"]

    if force_method is None:
        force_method = "random"

    force_method = str(force_method).lower().strip()

    if force_method == "random":
        chosen = methods[int(rng.integers(0, len(methods)))]
    else:
        if force_method not in methods:
            raise ValueError(f"Invalid force_method: {force_method}. Options: {methods} or 'random'")
        chosen = force_method

    if chosen == "gauss":
        alpha, p = _alpha_gaussian(mask_u8, rng)
        return alpha, "gauss", p
    elif chosen == "dist":
        alpha, p = _alpha_distance_feather(mask_u8, rng)
        return alpha, "dist", p
    else:
        alpha, p = _alpha_morphological(mask_u8, rng)
        return alpha, "morph", p


# -----------------------------
# Overlay (baseline + debug params)
# -----------------------------
def soft_overlay_on_scene(scene_bgr,
                          tag_crop_bgr,
                          mask_crop,
                          center_xy,
                          rng: np.random.Generator,
                          alpha_force: str = "random"):

    """
    Returns six outputs compatible with the main script:
        scene, bbox, alpha_method_with_params, blend_mode, shadow_on, occ_on
    """
    scene = scene_bgr.copy()
    h_tag, w_tag = tag_crop_bgr.shape[:2]
    h_s, w_s = scene.shape[:2]

    cx, cy = center_xy
    x0 = int(cx - w_tag / 2)
    y0 = int(cy - h_tag / 2)
    x1 = x0 + w_tag
    y1 = y0 + h_tag

    if x0 < 0 or y0 < 0 or x1 > w_s or y1 > h_s:
        raise ValueError("Tag extends outside the scene; adjust center_xy or size.")

    # Adapt the tag to the ROI profile
    roi_box = (x0, y0, x1, y1)
    tag_crop_bgr = adapt_tag_to_scene_roi(tag_crop_bgr, scene_bgr, roi_box, rng=rng)

    roi = scene[y0:y1, x0:x1]

    # Alpha stage: method and parameter debug
    # alpha, alpha_method, alpha_params = build_alpha(mask_crop, rng)
    alpha, alpha_method, alpha_params = build_alpha(mask_crop, rng, force_method=alpha_force)


    tag_f = tag_crop_bgr.astype(np.float32)
    roi_f = roi.astype(np.float32)

    # Halo/glare handling:
    base = (mask_crop.astype(np.float32))[..., None]  # 0/1
    tag_f = tag_f * base + roi_f * (1.0 - base)

    out_f = alpha * tag_f + (1.0 - alpha) * roi_f
    roi[:] = np.clip(out_f, 0, 255).astype(np.uint8)

    bbox = (x0, y0, x1, y1)

    # Baseline: alpha blending only
    blend_mode = "alpha"
    shadow_on = False
    occ_on = False

    # Combine into one string for reporting in the main script
    alpha_method_with_params = f"{alpha_method}({alpha_params})"

    return scene, bbox, alpha_method_with_params, blend_mode, shadow_on, occ_on


def embed_tag_on_scene(scene_bgr,
                       tag_path,
                       target_size_px,
                       center_xy,
                       yaw_deg=0.0,
                       pitch_deg=0.0,
                       roll_deg=0.0,
                       fov_deg=60.0,
                       dist_scale=3.0,
                       bg_color=(200, 200, 200),
                       rng=None, 
                       alpha_force: str = "random"):
    """
    Returns six outputs compatible with the main script:
        composed, bbox, alpha_method_with_params, blend_mode, shadow_on, occ_on
    """
    if rng is None:
        rng = np.random.default_rng()

    tag_img = cv2.imread(tag_path)
    if tag_img is None:
        raise FileNotFoundError(f"Tag image could not be read: {tag_path}")

    tag_resized = cv2.resize(tag_img, (target_size_px, target_size_px), interpolation=cv2.INTER_AREA)

    tag_crop, mask_crop = warp_and_crop_with_mask(
        tag_resized,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        fov_deg=fov_deg,
        dist_scale=dist_scale,
        border_color=bg_color
    )

    return soft_overlay_on_scene(
        scene_bgr,
        tag_crop,
        mask_crop,
        center_xy=center_xy,
        rng=rng,
        alpha_force=alpha_force
    )
