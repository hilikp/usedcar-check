from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class UnderbodyFinding:
    label: str
    confidence: float
    details: dict


def analyze_underbody_image(path: str) -> list[UnderbodyFinding]:
    """
    Heuristic detector for actual fluid stains on underbody images.
    Uses two complementary signals:
      1. Oil-colour blobs — brownish/amber tones (HSV hue 8-25, medium saturation)
         which are characteristic of engine oil or gear oil stains.
      2. Large contiguous dark-wet patches that are also locally concentrated
         (not just general dark background from metal/undercoating).
    Returns a finding only when there is clear visual evidence, not just darkness.
    """
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]   # 0-179
    s = hsv[:, :, 1]   # 0-255
    v = hsv[:, :, 2]   # 0-255

    findings: list[UnderbodyFinding] = []

    # ── Signal 1: Oil-stain colour (brownish/amber) ─────────────────────────
    # Hue 8-22 (brown/amber in OpenCV 0-179 scale), moderate saturation,
    # not too dark (visible stain, not just shadow).
    oil_mask = (h >= 8) & (h <= 22) & (s >= 50) & (s <= 200) & (v >= 30) & (v <= 160)
    oil_ratio = float(oil_mask.mean())

    # Require a contiguous blob for oil colour signal
    oil_blob_ok = False
    if oil_ratio > 0.02:
        oil_u8 = oil_mask.astype(np.uint8) * 255
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(oil_u8, connectivity=8)
        # Largest non-background blob
        if num_labels > 1:
            largest = int(stats[1:, cv2.CC_STAT_AREA].max())
            total_px = img.shape[0] * img.shape[1]
            if largest / total_px > 0.030:   # blob covers >3% of image
                oil_blob_ok = True

    if oil_blob_ok:
        confidence = min(0.72, 0.40 + oil_ratio * 3)
        findings.append(
            UnderbodyFinding(
                label="possible_underbody_fluid_stain",
                confidence=confidence,
                details={"signal": "oil_colour", "oil_ratio": round(oil_ratio, 3)},
            )
        )
        return findings   # strong signal — no need to check further

    # ── Signal 2: Concentrated dark-wet patches ──────────────────────────────
    # Dark (v < 45) AND very low saturation (s < 50) — a wet oily surface
    # looks uniformly very dark and loses colour.  We require:
    #   a) The ratio of such pixels exceeds 0.40 (very dominant — not just shadows)
    #   b) AND the image contains a meaningful bright region too (i.e. the photo
    #      was taken with light; a uniformly dark photo is just bad lighting).
    dark_mask = (v < 45) & (s < 50)
    dark_ratio = float(dark_mask.mean())
    bright_ratio = float((v > 120).mean())

    if dark_ratio > 0.55 and bright_ratio > 0.08:
        # Additionally require a single large contiguous dark blob
        dark_u8 = dark_mask.astype(np.uint8) * 255
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dark_u8, connectivity=8)
        if num_labels > 1:
            largest = int(stats[1:, cv2.CC_STAT_AREA].max())
            total_px = img.shape[0] * img.shape[1]
            if largest / total_px > 0.20:   # blob covers >20% of image
                confidence = min(0.60, 0.30 + dark_ratio * 0.5)
                findings.append(
                    UnderbodyFinding(
                        label="possible_underbody_fluid_stain",
                        confidence=confidence,
                        details={"signal": "dark_patch", "dark_ratio": round(dark_ratio, 3)},
                    )
                )

    return findings
