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

    Signal 1a — Oil colour (brownish/amber hue 8-22): specific to oil stains.
    Signal 1b — Coolant colour (green hue 35-85): specific to antifreeze.
    Signal 2  — Large dark-wet patch: very conservative thresholds to avoid
                false-positives from normal dark undercoating/shadows.
    Signal 3  — Shiny reflective patch: high threshold to avoid false-positives
                from clean metal / reflections on a new car.
    """
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]   # 0-179
    s = hsv[:, :, 1]   # 0-255
    v = hsv[:, :, 2]   # 0-255

    findings: list[UnderbodyFinding] = []
    total_px = img.shape[0] * img.shape[1]

    # ── Signal 1a: Oil stain (brownish/amber) ───────────────────────────────
    # Hue 8-22 (brown/amber), moderate saturation, not too dark.
    oil_mask  = (h >= 8) & (h <= 22) & (s >= 35) & (s <= 220) & (v >= 20) & (v <= 180)
    oil_ratio = float(oil_mask.mean())

    if oil_ratio > 0.008:
        oil_u8 = oil_mask.astype(np.uint8) * 255
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(oil_u8, connectivity=8)
        if num_labels > 1:
            largest = int(stats[1:, cv2.CC_STAT_AREA].max())
            if largest / total_px > 0.008:
                confidence = min(0.85, 0.55 + oil_ratio * 4)
                findings.append(UnderbodyFinding(
                    label="possible_underbody_fluid_stain",
                    confidence=confidence,
                    details={"signal": "oil_colour", "oil_ratio": round(oil_ratio, 3)},
                ))
                return findings

    # ── Signal 1b: Coolant / antifreeze (green) ─────────────────────────────
    coolant_mask  = (h >= 35) & (h <= 85) & (s >= 40) & (v >= 25) & (v <= 180)
    coolant_ratio = float(coolant_mask.mean())

    if coolant_ratio > 0.008:
        cool_u8 = coolant_mask.astype(np.uint8) * 255
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(cool_u8, connectivity=8)
        if num_labels > 1:
            largest = int(stats[1:, cv2.CC_STAT_AREA].max())
            if largest / total_px > 0.008:
                confidence = min(0.80, 0.50 + coolant_ratio * 4)
                findings.append(UnderbodyFinding(
                    label="possible_underbody_fluid_stain",
                    confidence=confidence,
                    details={"signal": "coolant_colour", "coolant_ratio": round(coolant_ratio, 3)},
                ))
                return findings

    # ── Signal 2: Concentrated dark-wet patch ───────────────────────────────
    # Wet oily surfaces look uniformly very dark and low-chroma.
    # Thresholds are deliberately conservative: normal dark undercoating and
    # shadows trigger v<55 easily — we need a VERY large concentrated area.
    dark_mask   = (v < 45) & (s < 45)         # tighter than before
    dark_ratio  = float(dark_mask.mean())
    bright_ratio = float((v > 120).mean())     # some bright area confirms contrast

    if dark_ratio > 0.60 and bright_ratio > 0.10:   # 60% dark + 10% bright
        dark_u8 = dark_mask.astype(np.uint8) * 255
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dark_u8, connectivity=8)
        if num_labels > 1:
            largest = int(stats[1:, cv2.CC_STAT_AREA].max())
            if largest / total_px > 0.30:       # blob must cover 30% of image
                confidence = min(0.70, 0.40 + dark_ratio * 0.5)
                findings.append(UnderbodyFinding(
                    label="possible_underbody_fluid_stain",
                    confidence=confidence,
                    details={"signal": "dark_patch", "dark_ratio": round(dark_ratio, 3)},
                ))
                return findings

    # ── Signal 3: Shiny / reflective wet patch ──────────────────────────────
    # Fresh fluid reflects light (high V, low-medium S).
    # Threshold raised significantly — clean metal and lighting on a new car
    # easily produce small shiny areas that are NOT leaks.
    shiny_mask  = (v > 200) & (s < 60)        # brighter and less saturated than before
    shiny_ratio = float(shiny_mask.mean())

    if shiny_ratio > 0.08:                     # 8% of image (was 0.5%)
        shiny_u8 = shiny_mask.astype(np.uint8) * 255
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(shiny_u8, connectivity=8)
        if num_labels > 1:
            largest = int(stats[1:, cv2.CC_STAT_AREA].max())
            if largest / total_px > 0.06:      # blob must cover 6% (was 0.5%)
                confidence = min(0.60, 0.30 + shiny_ratio * 3)
                findings.append(UnderbodyFinding(
                    label="possible_underbody_fluid_stain",
                    confidence=confidence,
                    details={"signal": "shiny_patch", "shiny_ratio": round(shiny_ratio, 3)},
                ))

    return findings
