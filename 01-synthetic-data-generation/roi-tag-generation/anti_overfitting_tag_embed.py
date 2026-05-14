# anti_overfitting_tag_embed.py
# -*- coding: utf-8 -*-
"""
Synthetic full-tag placement module for ROI detector training.

The module applies background-aware filtering, appearance variation,
adaptive blending, and bounding-box generation for full-tag samples.
"""

import cv2
import numpy as np
from background_profile import compute_patch_profile


# ============================================
# BLENDING METHODS
# ============================================

def blend_normal(tag_f, roi_f, alpha):
    """Standard alpha blending."""
    return alpha * tag_f + (1.0 - alpha) * roi_f


def blend_multiply(tag_f, roi_f, alpha):
    """Multiply blending, with darker tones emphasized."""
    mult = (tag_f / 255.0) * (roi_f / 255.0) * 255.0
    return alpha * mult + (1.0 - alpha) * roi_f


def blend_screen(tag_f, roi_f, alpha):
    """Screen blending, with brighter tones emphasized."""
    screen = 255.0 - ((255.0 - tag_f) / 255.0) * ((255.0 - roi_f) / 255.0) * 255.0
    return alpha * screen + (1.0 - alpha) * roi_f


def blend_overlay(tag_f, roi_f, alpha):
    """Overlay blending for local contrast enhancement."""
    mask = roi_f < 128
    result = np.zeros_like(tag_f)
    
    # Dark regions: multiply
    result[mask] = 2.0 * tag_f[mask] * roi_f[mask] / 255.0
    
    # Bright regions: screen
    result[~mask] = 255.0 - 2.0 * (255.0 - tag_f[~mask]) * (255.0 - roi_f[~mask]) / 255.0
    
    return alpha * result + (1.0 - alpha) * roi_f


def blend_soft_light(tag_f, roi_f, alpha):
    """Soft-light blending."""
    t_norm = tag_f / 255.0
    r_norm = roi_f / 255.0
    
    result = ((1.0 - 2.0 * t_norm) * r_norm * r_norm + 
              2.0 * t_norm * r_norm) * 255.0
    
    return alpha * result + (1.0 - alpha) * roi_f


def blend_hard_light(tag_f, roi_f, alpha):
    """Hard-light blending."""
    mask = tag_f < 128
    result = np.zeros_like(tag_f)
    
    result[mask] = 2.0 * tag_f[mask] * roi_f[mask] / 255.0
    result[~mask] = 255.0 - 2.0 * (255.0 - tag_f[~mask]) * (255.0 - roi_f[~mask]) / 255.0
    
    return alpha * result + (1.0 - alpha) * roi_f


def blend_darken(tag_f, roi_f, alpha):
    """Darken mode: select the darker value for each pixel."""
    darkened = np.minimum(tag_f, roi_f)
    return alpha * darkened + (1.0 - alpha) * roi_f


def blend_lighten(tag_f, roi_f, alpha):
    """Lighten mode: select the brighter value for each pixel."""
    lightened = np.maximum(tag_f, roi_f)
    return alpha * lightened + (1.0 - alpha) * roi_f


BLEND_METHODS = {
    "normal": blend_normal,
    "multiply": blend_multiply,
    "screen": blend_screen,
    "overlay": blend_overlay,
    "soft_light": blend_soft_light,
    "hard_light": blend_hard_light,
    "darken": blend_darken,
    "lighten": blend_lighten,
}

PIPELINE_MODE_FULL = "full"
PIPELINE_MODE_NAIVE = "naive_same_scale"
PIPELINE_MODE_NO_BG_AWARE = "no_bg_aware_block"
PIPELINE_MODE_NO_APPEARANCE = "no_appearance_realism_block"
VALID_PIPELINE_MODES = {
    PIPELINE_MODE_FULL,
    PIPELINE_MODE_NAIVE,
    PIPELINE_MODE_NO_BG_AWARE,
    PIPELINE_MODE_NO_APPEARANCE,
}


def _resolve_pipeline_flags(pipeline_mode: str):
    mode = str(pipeline_mode).strip().lower()
    if mode not in VALID_PIPELINE_MODES:
        raise ValueError(
            f"Invalid pipeline_mode: {pipeline_mode}. "
            f"Options: {sorted(VALID_PIPELINE_MODES)}"
        )

    flags = {
        "mode": mode,
        "use_paper_texture": False,
        "use_photometric_profile": False,
        "use_appearance_realism": False,
        "use_alpha_builder": False,
        "use_mask_gate": False,
        "use_adaptive_blending": False,
        "use_occlusion": False,
        "use_post_jitter": False,
    }

    if mode == PIPELINE_MODE_FULL:
        flags.update({
            "use_paper_texture": True,
            "use_photometric_profile": True,
            "use_appearance_realism": True,
            "use_alpha_builder": True,
            "use_mask_gate": True,
            "use_adaptive_blending": True,
            "use_occlusion": True,
            "use_post_jitter": True,
        })
    elif mode == PIPELINE_MODE_NAIVE:
        flags.update({
            # Naive same-scale: raw copy-paste placement used for ablation.
            # Advanced adaptations are disabled; mask and alpha handling are also off.
        })
    elif mode == PIPELINE_MODE_NO_BG_AWARE:
        flags.update({
            # Background-aware block is disabled; basic masked placement remains active.
            "use_appearance_realism": True,
            "use_alpha_builder": True,
            "use_mask_gate": True,
            "use_occlusion": True,
            "use_post_jitter": True,
        })
    elif mode == PIPELINE_MODE_NO_APPEARANCE:
        flags.update({
            "use_paper_texture": True,
            "use_photometric_profile": True,
            "use_alpha_builder": True,
            "use_mask_gate": True,
            "use_adaptive_blending": True,
        })

    return flags



def add_opaque_background_to_tag(tag_bgr, mask_crop, rng):
    """
    Add a realistic opaque paper background to transparent tag crops.
    Printed tags are typically placed on white or cream paper.

    The mask is expanded so that the full tag crop becomes opaque.
    """
    # Select a realistic paper background color.
    bg_type = rng.choice(["white", "cream", "light_gray", "yellow"], 
                        p=[0.50, 0.25, 0.15, 0.10])
    
    if bg_type == "white":
        base_color = np.array([245, 245, 245]) + rng.uniform(-10, 10, 3)
    elif bg_type == "cream":
        base_color = np.array([230, 240, 250]) + rng.uniform(-15, 10, 3)
    elif bg_type == "light_gray":
        base_color = np.array([220, 220, 220]) + rng.uniform(-20, 15, 3)
    else:  # yellow
        base_color = np.array([200, 230, 255]) + rng.uniform(-10, 10, 3)
    
    base_color = np.clip(base_color, 200, 255)  # Light tones
    
    # Add a light paper texture.
    h, w = tag_bgr.shape[:2]
    texture_noise = rng.normal(0, 3, (h, w, 3))
    
    # Create the background layer.
    background = np.ones((h, w, 3), dtype=np.float32) * base_color
    background = background + texture_noise
    background = np.clip(background, 0, 255)
    
    # Dilate the mask so the tag edges are fully covered.
    mask_binary = (mask_crop > 0.1).astype(np.uint8)
    kernel_size = int(rng.integers(3, 8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask_expanded = cv2.dilate(mask_binary, kernel, iterations=1)
    mask_expanded = mask_expanded.astype(np.float32)
    
    # Apply a light blur because paper edges are not perfectly sharp.
    blur_size = int(rng.integers(3, 7))
    if blur_size % 2 == 0:
        blur_size += 1
    mask_expanded = cv2.GaussianBlur(mask_expanded, (blur_size, blur_size), 0)
    
    mask_3ch = mask_expanded[:, :, None]
    
    # Tag content with an opaque background.
    # Keep the background across the whole crop, not only inside the tag area.
    result = tag_bgr.astype(np.float32) * mask_3ch + background * (1.0 - mask_3ch)
    
    # Make the full crop opaque so the borders retain the paper color.
    result = result.astype(np.uint8)
    
    # Return the expanded mask.
    return result, mask_expanded, bg_type


def validate_tag_crop_quality(tag_crop, mask_crop, min_area_ratio=0.25, max_area_ratio=0.95):
    """
    Check whether the tag crop is suitable for object-detection training.

    Args:
        tag_crop: Cropped tag image.
        mask_crop: Tag mask.
        min_area_ratio: Minimum area ratio used to reject very small tags.
        max_area_ratio: Maximum area ratio for near-full crop coverage.

    Returns:
        (is_valid, reason, metrics)
    """
    h, w = tag_crop.shape[:2]
    total_pixels = h * w
    
    # 1. Area check
    mask_binary = (mask_crop > 0.3).astype(np.uint8)  # Lower threshold
    tag_pixels = np.sum(mask_binary)
    area_ratio = tag_pixels / total_pixels
    
    # Reject very small tags.
    if area_ratio < min_area_ratio:
        return False, f"area_too_small_{area_ratio:.2f}", {"area_ratio": area_ratio}
    
    # 2. Minimum crop-size check in pixels
    if h < 25 or w < 25:
        return False, f"crop_too_small_{w}x{h}", {"area_ratio": area_ratio, "crop_size": (w, h)}
    
    # 3. Contour check
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return False, "no_contours", {"area_ratio": area_ratio}
    
    # Use the largest contour.
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w_box, h_box = cv2.boundingRect(main_contour)
    
    # 4. Minimum bounding-box size check
    if w_box < 25 or h_box < 25:
        return False, f"bbox_too_small_{w_box}x{h_box}", {
            "area_ratio": area_ratio,
            "bbox_size": (w_box, h_box)
        }
    
    # 5. Aspect-ratio check with a relaxed limit
    aspect_ratio = max(w_box, h_box) / (min(w_box, h_box) + 1e-6)
    
    # Very elongated crops (>6:1) are usually poor samples.
    if aspect_ratio > 6.0:
        return False, f"bad_aspect_{aspect_ratio:.1f}", {
            "area_ratio": area_ratio, 
            "aspect_ratio": aspect_ratio
        }
    
    # 6. Fragmentation check with a relaxed limit
    if len(contours) > 2:  # Check only when more than two components exist.
        # The largest component should cover more than 70% of the total area.
        main_area = cv2.contourArea(main_contour)
        if main_area / tag_pixels < 0.70:
            return False, "fragmented", {
                "area_ratio": area_ratio,
                "fragmentation": main_area / tag_pixels
            }
    
    # All checks passed.
    return True, "valid", {
        "area_ratio": area_ratio,
        "aspect_ratio": aspect_ratio,
        "bbox_size": (w_box, h_box)
    }


def compute_tight_bbox_from_mask(mask_crop, padding_ratio=0.05):
    """
    Compute a tight bounding box from the mask.
    A small padding is added to produce a practical object-detection box.

    Args:
        mask_crop: Tag mask in the 0-1 range.
        padding_ratio: Padding ratio added around the mask.

    Returns:
        (x_min, y_min, x_max, y_max) in non-normalized pixel coordinates.
    """
    mask_binary = (mask_crop > 0.5).astype(np.uint8)
    
    # Find contours.
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        # Fallback: use the full mask.
        h, w = mask_crop.shape[:2]
        return (0, 0, w, h)
    
    # Compute the bounding box from the largest contour.
    main_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(main_contour)
    
    # Add a small padding because manual boxes are often slightly wider.
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    
    x_min = max(0, x - pad_x)
    y_min = max(0, y - pad_y)
    x_max = min(mask_crop.shape[1], x + w + pad_x)
    y_max = min(mask_crop.shape[0], y + h + pad_y)
    
    return (x_min, y_min, x_max, y_max)


def add_bbox_augmentation(bbox, scene_shape, rng, augment_prob=0.3):
    """
    Apply light augmentation to the bounding box.
    This simulates small annotation differences in practical datasets.

    Args:
        bbox: (x0, y0, x1, y1) in scene coordinates.
        scene_shape: (H, W, C).
        rng: Random generator.
        augment_prob: Probability of applying augmentation.

    Returns:
        Augmented bounding box.
    """
    if rng.random() > augment_prob:
        return bbox
    
    x0, y0, x1, y1 = bbox
    h_s, w_s = scene_shape[:2]
    
    w_box = x1 - x0
    h_box = y1 - y0
    
    # Light jitter in the 2-5% range.
    jitter_ratio = rng.uniform(0.02, 0.05)
    
    jitter_x = int(w_box * jitter_ratio * rng.choice([-1, 1]))
    jitter_y = int(h_box * jitter_ratio * rng.choice([-1, 1]))
    
    # Apply it to random box edges.
    if rng.random() < 0.5:
        x0 = max(0, x0 + jitter_x)
    if rng.random() < 0.5:
        x1 = min(w_s, x1 + jitter_x)
    if rng.random() < 0.5:
        y0 = max(0, y0 + jitter_y)
    if rng.random() < 0.5:
        y1 = min(h_s, y1 + jitter_y)
    
    # Minimum size check.
    if x1 - x0 < 15 or y1 - y0 < 15:
        return bbox  # Revert the jitter.
    
    return (x0, y0, x1, y1)


# ============================================
# 1. REAL-WORLD VARIATIONS
# ============================================

def apply_realistic_degradation(tag_bgr, rng, severity="random"):
    """
    Apply degradations that printed tags may undergo in real scenes:
    - wear, fading, dirt, and local stains.
    """
    if severity == "random":
        severity = rng.choice(["none", "light", "medium", "heavy"], 
                             p=[0.3, 0.4, 0.2, 0.1])
    
    if severity == "none":
        return tag_bgr, "none"
    
    tag_f = tag_bgr.astype(np.float32)
    h, w = tag_f.shape[:2]
    
    applied_effects = []
    
    # 1. Color fading from UV exposure
    if rng.random() < 0.6:
        fade_amount = rng.uniform(0.05, 0.25) if severity != "heavy" else rng.uniform(0.2, 0.4)
        lab = cv2.cvtColor(tag_f.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        L, a, b = cv2.split(lab)
        
        # Color desaturation
        a = a * (1.0 - fade_amount) + 128 * fade_amount
        b = b * (1.0 - fade_amount) + 128 * fade_amount
        
        # Brightness reduction
        L = L * (1.0 - fade_amount * 0.3)
        
        lab = cv2.merge([L, a, b])
        tag_f = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        applied_effects.append("fade")
    
    # 2. Dirt overlay
    if rng.random() < 0.4:
        dirt_intensity = rng.uniform(0.05, 0.15) if severity != "heavy" else rng.uniform(0.15, 0.35)
        
        # Perlin-like noise pattern
        dirt_freq = rng.uniform(0.01, 0.05)
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        dirt_noise = np.sin(x * dirt_freq) * np.cos(y * dirt_freq * 0.7)
        dirt_noise = (dirt_noise + 1.0) / 2.0  # Normalize to 0-1.
        
        # Gaussian blur
        dirt_noise = cv2.GaussianBlur(dirt_noise, (15, 15), 0)
        
        # Dirt color in brownish tones
        dirt_color = rng.uniform(0.3, 0.6, size=3) * 255
        dirt_layer = dirt_noise[:, :, None] * dirt_color * dirt_intensity
        
        tag_f = tag_f * (1.0 - dirt_intensity * dirt_noise[:, :, None]) + dirt_layer
        applied_effects.append("dirt")
    
    # 3. Scratches
    if rng.random() < 0.3 and severity in ["medium", "heavy"]:
        num_scratches = int(rng.integers(2, 8 if severity == "heavy" else 4))
        
        for _ in range(num_scratches):
            # Random line
            pt1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            pt2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            thickness = int(rng.integers(1, 3))
            
            # Scratch color, usually light or dark
            scratch_val = rng.uniform(0.3, 0.7)
            color = tuple([scratch_val * 255] * 3)
            
            cv2.line(tag_f.astype(np.uint8), pt1, pt2, color, thickness)
        
        applied_effects.append("scratch")
    
    # 4. Local color stains
    if rng.random() < 0.35:
        num_stains = int(rng.integers(1, 4))
        
        for _ in range(num_stains):
            cx = int(rng.integers(0, w))
            cy = int(rng.integers(0, h))
            radius = int(rng.integers(10, min(w, h) // 3))
            
            stain_color = rng.uniform(0.4, 0.9, size=3) * 255
            stain_intensity = rng.uniform(0.1, 0.3)
            
            # Soft edge
            Y, X = np.ogrid[:h, :w]
            dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
            mask = np.clip(1.0 - dist / radius, 0, 1)
            mask = cv2.GaussianBlur(mask.astype(np.float32), (15, 15), 0)
            
            stain = mask[:, :, None] * stain_color * stain_intensity
            tag_f = tag_f * (1.0 - mask[:, :, None] * stain_intensity) + stain
        
        applied_effects.append("stain")
    
    result = np.clip(tag_f, 0, 255).astype(np.uint8)
    effect_str = "+".join(applied_effects) if applied_effects else "none"
    
    return result, effect_str


def apply_environmental_lighting(tag_bgr, scene_roi, rng, strength_range=(0.3, 0.8)):
    """
    Adapt the tag to the surrounding illumination.
    Printed tags do not act as independent light sources in real scenes.
    """
    strength = float(rng.uniform(*strength_range))
    
    # Estimate the scene color temperature.
    lab_scene = cv2.cvtColor(scene_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_scene, a_scene, b_scene = cv2.split(lab_scene)
    
    scene_avg_L = float(np.mean(L_scene))
    scene_avg_a = float(np.mean(a_scene))
    scene_avg_b = float(np.mean(b_scene))
    
    # Convert the tag to LAB.
    lab_tag = cv2.cvtColor(tag_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    L_tag, a_tag, b_tag = cv2.split(lab_tag)
    
    # Apply scene illumination.
    # 1. Brightness adaptation
    L_ratio = scene_avg_L / (np.mean(L_tag) + 1e-6)
    L_ratio = np.clip(L_ratio, 0.5, 2.0)
    L_adapted = L_tag * (1.0 + strength * (L_ratio - 1.0))
    
    # 2. Color-temperature adaptation
    a_shift = (scene_avg_a - np.mean(a_tag)) * strength * 0.6
    b_shift = (scene_avg_b - np.mean(b_tag)) * strength * 0.6
    
    a_adapted = a_tag + a_shift
    b_adapted = b_tag + b_shift
    
    # 3. Local illumination variation from the scene gradient
    L_scene_normalized = (L_scene - L_scene.min()) / (L_scene.max() - L_scene.min() + 1e-6)
    
    # Modulate the tag using the scene illumination map.
    h_tag, w_tag = L_tag.shape
    h_scene, w_scene = L_scene.shape
    
    if h_tag <= h_scene and w_tag <= w_scene:
        # Resize the illumination pattern to the tag size.
        local_light = cv2.resize(L_scene_normalized, (w_tag, h_tag), 
                                interpolation=cv2.INTER_LINEAR)
        
        # Apply local illumination variation.
        light_variation = (local_light - 0.5) * 50.0 * strength * 0.4
        L_adapted = L_adapted + light_variation
    
    # Merge the channels.
    lab_result = cv2.merge([L_adapted, a_adapted, b_adapted])
    result = cv2.cvtColor(np.clip(lab_result, 0, 255).astype(np.uint8), 
                         cv2.COLOR_LAB2BGR)
    
    return result


def add_realistic_occlusion(scene_roi, tag_mask, rng):
    """
    Add conservative edge occlusion effects.

    The occlusion should affect only a small part of the tag, mainly from the edges.
    """
    h, w = tag_mask.shape[:2]
    occlusion_mask = np.ones((h, w), dtype=np.float32)
    occlusion_applied = False
    
    # Use a conservative occlusion probability and intensity.
    if rng.random() < 0.25:  # 25% probability
        edge_type = rng.choice(["corner", "side"])  # Irregular occlusion disabled.
        
        if edge_type == "corner":
            # Smaller corner wear
            corner = rng.choice(["tl", "tr", "bl", "br"])
            size = int(rng.uniform(0.08, 0.18) * min(h, w))  # Previously 0.15-0.35
            
            Y, X = np.mgrid[0:h, 0:w]
            if corner == "tl":
                dist = np.sqrt(X**2 + Y**2)
            elif corner == "tr":
                dist = np.sqrt((w - X)**2 + Y**2)
            elif corner == "bl":
                dist = np.sqrt(X**2 + (h - Y)**2)
            else:  # br
                dist = np.sqrt((w - X)**2 + (h - Y)**2)
            
            fade = np.clip(dist / size, 0, 1)
            occlusion_mask *= fade
            occlusion_applied = True
        
        elif edge_type == "side":
            # Shallower side-edge wear
            side = rng.choice(["top", "bottom", "left", "right"])
            depth = int(rng.uniform(0.05, 0.15) * (h if side in ["top", "bottom"] else w))  # Previously 0.1-0.3
            
            Y, X = np.mgrid[0:h, 0:w]
            if side == "top":
                fade = np.clip(Y / depth, 0, 1)
            elif side == "bottom":
                fade = np.clip((h - Y) / depth, 0, 1)
            elif side == "left":
                fade = np.clip(X / depth, 0, 1)
            else:  # right
                fade = np.clip((w - X) / depth, 0, 1)
            
            occlusion_mask *= fade
            occlusion_applied = True
    
    # Random partial occlusion was removed because it was too aggressive.
    
    return occlusion_mask[:, :, None], occlusion_applied


def add_perspective_depth_cues(tag_bgr, mask, perspective_angle, rng):
    """
    Add perspective-depth cues:
    - distant regions become slightly blurred,
    - distant regions become darker through atmospheric perspective.
    """
    h, w = tag_bgr.shape[:2]
    
    # Build the depth gradient according to the perspective direction.
    if abs(perspective_angle[0]) > abs(perspective_angle[1]):  # Yaw dominant
        # Horizontal gradient
        if perspective_angle[0] > 0:
            gradient = np.linspace(1.0, 0.7, w)
            gradient = np.tile(gradient[None, :], (h, 1))
        else:
            gradient = np.linspace(0.7, 1.0, w)
            gradient = np.tile(gradient[None, :], (h, 1))
    else:  # Pitch dominant
        # Vertical gradient
        if perspective_angle[1] > 0:
            gradient = np.linspace(1.0, 0.7, h)
            gradient = np.tile(gradient[:, None], (1, w))
        else:
            gradient = np.linspace(0.7, 1.0, h)
            gradient = np.tile(gradient[:, None], (1, w))
    
    # Gradual blurring
    blur_strength = abs(perspective_angle[0]) + abs(perspective_angle[1])
    if blur_strength > 20:  # Only for clear perspective changes
        # Multi-level blur for distant regions
        blurred_versions = [tag_bgr]
        for blur_size in [3, 5, 7]:
            blurred = cv2.GaussianBlur(tag_bgr, (blur_size, blur_size), 0)
            blurred_versions.append(blurred)
        
        # Blend according to the gradient.
        result = np.zeros_like(tag_bgr, dtype=np.float32)
        grad_levels = np.clip((1.0 - gradient) * 3, 0, 3)  # Range: 0-3
        
        for i in range(h):
            for j in range(w):
                level = grad_levels[i, j]
                idx = int(level)
                frac = level - idx
                
                if idx >= len(blurred_versions) - 1:
                    result[i, j] = blurred_versions[-1][i, j]
                else:
                    result[i, j] = (blurred_versions[idx][i, j] * (1 - frac) + 
                                  blurred_versions[idx + 1][i, j] * frac)
        
        tag_bgr = result.astype(np.uint8)
    
    # Atmospheric perspective: distant regions are slightly darker or hazier.
    darkness = gradient * 0.85 + 0.15  # Range: 0.15-1.0
    result = tag_bgr.astype(np.float32) * darkness[:, :, None]
    
    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================
# 2. ADVANCED BLENDING ENGINE
# ============================================

def adaptive_blend_selection(tag_prof, roi_prof, rng):
    """
    Select the blending mode according to the tag and ROI profiles.
    This approximates material-dependent appearance changes.
    """
    # Brightness difference
    brightness_diff = abs(tag_prof.mean_L - roi_prof.mean_L)
    
    # Contrast level
    contrast_level = max(tag_prof.std_L, roi_prof.std_L)
    
    # Weight the candidate modes.
    if brightness_diff < 30:
        # Similar brightness: prefer normal or soft-light blending.
        weights = {
            "normal": 0.50,
            "soft_light": 0.25,
            "overlay": 0.15,
            "multiply": 0.05,
            "screen": 0.05,
        }
    elif tag_prof.mean_L > roi_prof.mean_L + 30:
        # Brighter tag: screen or lighten can be selected.
        weights = {
            "normal": 0.35,
            "screen": 0.30,
            "lighten": 0.20,
            "soft_light": 0.15,
        }
    else:
        # Darker tag: multiply or darken can be selected.
        weights = {
            "normal": 0.35,
            "multiply": 0.30,
            "darken": 0.20,
            "soft_light": 0.15,
        }
        

    # ----------------------------------------------------
    # Matte A4 paper tag: soft-light and lighten control
    # ----------------------------------------------------
    roi_bright = roi_prof.mean_L > 175  # Bright ROI: lighten may suppress the tag.
    tag_paperish = (tag_prof.mean_L > 190) and (tag_prof.std_L < 18)  # Very bright and low contrast

    if roi_bright:
        # On bright backgrounds, lighten may favor the ROI and weaken the tag.
        if "lighten" in weights:
            weights["lighten"] *= 0.03   # Nearly disable it.
        if "soft_light" in weights:
            weights["soft_light"] *= 0.35

    if tag_paperish:
        # If the paper is too bright, soft-light may have little effect and lighten is unnecessary.
        if "lighten" in weights:
            weights["lighten"] *= 0.05
        if "soft_light" in weights:
            weights["soft_light"] *= 0.55

    # Fallback: keep normal blending if all weights approach zero.
    if sum(weights.values()) < 1e-6:
        weights = {"normal": 1.0}
        
        
        
        
    
    # Add overlay or hard-light under high contrast.
    if contrast_level > 40:
        if "overlay" in weights:
            weights["overlay"] *= 1.5
        if "hard_light" not in weights:
            weights["hard_light"] = 0.1
    
    # Normalize
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}
    
    return rng.choice(list(weights.keys()), p=list(weights.values()))


# ============================================
# 3. MAIN SYSTEM
# ============================================

def anti_overfitting_overlay(scene_bgr,
                             tag_crop_bgr,
                             mask_crop,
                             center_xy,
                             perspective_angles,
                             rng: np.random.Generator,
                             alpha_force: str = "random",
                             pipeline_mode: str = "full"):
    """
    Mode-based overlay system.

    pipeline_mode:
        - full
        - naive_same_scale
        - no_bg_aware_block
        - no_appearance_realism_block

    Returns:
        scene, bbox, metadata_dict
    """
    from tag_embed_module import build_alpha

    flags = _resolve_pipeline_flags(pipeline_mode)

    # 0. Quality check: is the tag crop suitable?
    is_valid, reason, metrics = validate_tag_crop_quality(tag_crop_bgr, mask_crop)
    if not is_valid:
        raise ValueError(f"Tag crop quality insufficient: {reason} | metrics: {metrics}")

    scene = scene_bgr.copy()
    h_tag, w_tag = tag_crop_bgr.shape[:2]
    h_s, w_s = scene.shape[:2]

    cx, cy = center_xy
    x0 = int(cx - w_tag / 2)
    y0 = int(cy - h_tag / 2)
    x1 = x0 + w_tag
    y1 = y0 + h_tag

    if x0 < 0 or y0 < 0 or x1 > w_s or y1 > h_s:
        raise ValueError("The tag extends outside the scene.")

    roi_box = (x0, y0, x1, y1)
    roi = scene[y0:y1, x0:x1]

    metadata = {
        "crop_quality": metrics,
        "pipeline_mode": flags["mode"],
    }

    current_tag = tag_crop_bgr.copy()
    current_mask = mask_crop.astype(np.float32)

    tag_prof = None
    roi_prof = None
    if flags["use_photometric_profile"]:
        tag_prof = compute_patch_profile(current_tag)
        roi_prof = compute_patch_profile(roi)

    # 1. PAPER / OPAQUE BACKGROUND
    if flags["use_paper_texture"]:
        current_tag, current_mask, bg_type = add_opaque_background_to_tag(current_tag, current_mask, rng)
        metadata["background"] = bg_type
    else:
        metadata["background"] = "off"

    # 2-4. APPEARANCE REALISM BLOCK
    degradation_type = "none"
    if flags["use_appearance_realism"]:
        current_tag, degradation_type = apply_realistic_degradation(current_tag, rng, severity="random")
        current_tag = apply_environmental_lighting(current_tag, roi, rng)
        current_tag = add_perspective_depth_cues(current_tag, current_mask, perspective_angles, rng)
    metadata["degradation"] = degradation_type

    # 5. ALPHA
    if flags["use_alpha_builder"]:
        alpha, alpha_method, alpha_params = build_alpha(current_mask, rng, force_method=alpha_force)
        metadata["alpha"] = f"{alpha_method}({alpha_params})"
    else:
        alpha = np.ones(current_mask.shape[:2], dtype=np.float32)[..., None]
        metadata["alpha"] = "fixed(1.0)"

    # 6. BLEND MODE
    if flags["use_adaptive_blending"] and tag_prof is not None and roi_prof is not None:
        blend_mode = adaptive_blend_selection(tag_prof, roi_prof, rng)
    else:
        blend_mode = "normal"
    metadata["blend"] = blend_mode

    tag_f = current_tag.astype(np.float32)
    roi_f = roi.astype(np.float32)

    # Mask gate: keep the warp mask outside the naive mode.
    base = current_mask.astype(np.float32)[..., None]
    if flags["use_mask_gate"]:
        tag_f = tag_f * base + roi_f * (1.0 - base)
        alpha = alpha * base

    # 9. OCCLUSION
    occ_applied = False
    alpha_with_occlusion = alpha
    if flags["use_occlusion"]:
        occlusion_mask, occ_applied = add_realistic_occlusion(roi, alpha, rng)
        alpha_with_occlusion = alpha * occlusion_mask
    metadata["occlusion"] = occ_applied

    # 10. BLENDING
    blend_func = BLEND_METHODS.get(blend_mode, BLEND_METHODS["normal"])
    out_f = blend_func(tag_f, roi_f, alpha_with_occlusion)

    # POST-JITTER
    if flags["use_post_jitter"] and rng.random() < 0.3:
        jitter_amount = rng.uniform(-5, 5)
        out_f = out_f + jitter_amount
        metadata["jitter"] = f"{jitter_amount:.1f}"

    roi[:] = np.clip(out_f, 0, 255).astype(np.uint8)

    # 11. BBOX COMPUTATION
    tight_bbox_crop = compute_tight_bbox_from_mask(current_mask, padding_ratio=0.05)
    bbox = (
        x0 + tight_bbox_crop[0],
        y0 + tight_bbox_crop[1],
        x0 + tight_bbox_crop[2],
        y0 + tight_bbox_crop[3]
    )

    # 12. BBOX AUGMENTATION
    bbox = add_bbox_augmentation(bbox, scene.shape, rng, augment_prob=0.2)
    metadata["bbox_type"] = "tight"

    return scene, bbox, metadata

def embed_tag_anti_overfitting(scene_bgr,
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
                               alpha_force: str = "random",
                               pipeline_mode: str = "full"):
    """
    Main integration function.

    Returns:
        composed, bbox, alpha_method, blend_mode, occlusion_on, degradation_type
    """
    from tag_embed_module import warp_and_crop_with_mask

    if rng is None:
        rng = np.random.default_rng()

    tag_img = cv2.imread(tag_path)
    if tag_img is None:
        raise FileNotFoundError(f"Tag image could not be read: {tag_path}")

    tag_resized = cv2.resize(tag_img, (target_size_px, target_size_px),
                            interpolation=cv2.INTER_AREA)

    tag_crop, mask_crop = warp_and_crop_with_mask(
        tag_resized,
        yaw_deg=yaw_deg,
        pitch_deg=pitch_deg,
        roll_deg=roll_deg,
        fov_deg=fov_deg,
        dist_scale=dist_scale,
        border_color=bg_color
    )

    scene, bbox, metadata = anti_overfitting_overlay(
        scene_bgr,
        tag_crop,
        mask_crop,
        center_xy=center_xy,
        perspective_angles=(yaw_deg, pitch_deg),
        rng=rng,
        alpha_force=alpha_force,
        pipeline_mode=pipeline_mode
    )

    return (
        scene,
        bbox,
        metadata.get("alpha", "unknown"),
        metadata.get("blend", "normal"),
        metadata.get("occlusion", False),
        metadata.get("degradation", "none"),
    )
