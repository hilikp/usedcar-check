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
    # ── Signal 1a: Classic oil stain (brownish/amber) ──────────────────────
    oil_mask = (h >= 8) & (h <= 22) & (s >= 35) & (s <= 220) & (v >= 20) & (v <= 180)
    oil_ratio = float(oil_mask.mean())

    oil_blob_ok = False
    if oil_ratio > 0.008:   # lowered from 0.02 — smaller stains count
        oil_u8 = oil_mask.astype(np.uint8) * 255
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(oil_u8, connectivity=8)
        if num_labels > 1:
            largest = int(stats[1:, cv2.CC_STAT_AREA].max())
            total_px = img.shape[0] * img.shape[1]
            if largest / total_px > 0.008:   # lowered from 0.03 — 0.8% blob is enough
                oil_blob_ok = True

    if oil_blob_ok:
        confidence = min(0.85, 0.55 + oil_ratio * 4)
        findings.append(
            UnderbodyFinding(
                label="possible_underbody_fluid_stain",
                confidence=confidence,
                details={"signal": "oil_colour", "oil_ratio": round(oil_ratio, 3)},
            )
        )
        return findings

    # ── Signal 1b: Dark greenish tones — coolant / antifreeze ──────────────
    # Coolant is often green/pink; hue 35-85 (green range), moderate saturation
    coolant_mask = (h >= 35) & (h <= 85) & (s >= 40) & (v >= 25) & (v <= 180)
    coolant_ratio = float(coolant_mask.mean())
    if coolant_ratio > 0.008:
        cool_u8 = coolant_mask.astype(np.uint8) * 255
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(cool_u8, connectivity=8)
        if num_labels > 1:
            largest = int(stats[1:, cv2.CC_STAT_AREA].max())
            total_px = img.shape[0] * img.shape[1]
            if largest / total_px > 0.008:
                confidence = min(0.80, 0.50 + coolant_ratio * 4)
                findings.append(
                    UnderbodyFinding(
                        label="possible_underbody_fluid_stain",
                        confidence=confidence,
                        details={"signal": "coolant_colour", "coolant_ratio": round(coolant_ratio, 3)},
                    )
                )
                return findings

    # ── Signal 2: Concentrated dark-wet patches ──────────────────────────────
    # Wet oily surfaces look uniformly very dark and lose colour.
    dark_mask = (v < 55) & (s < 60)   # loosened from v<45, s<50
    dark_ratio = float(dark_mask.mean())
    bright_ratio = float((v > 100).mean())   # lowered from 120

    if dark_ratio > 0.35 and bright_ratio > 0.05:   # lowered from 0.55 / 0.08
        dark_u8 = dark_mask.astype(np.uint8) * 255
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(dark_u8, connectivity=8)
        if num_labels > 1:
            largest = int(stats[1:, cv2.CC_STAT_AREA].max())
            total_px = img.shape[0] * img.shape[1]
            if largest / total_px > 0.10:   # lowered from 0.20
                confidence = min(0.72, 0.40 + dark_ratio * 0.6)
                findings.append(
                    UnderbodyFinding(
                        label="possible_underbody_fluid_stain",
                        confidence=confidence,
                        details={"signal": "dark_patch", "dark_ratio": round(dark_ratio, 3)},
                    )
                )
                return findings

    # ── Signal 3: Shiny/reflective wet patch ─────────────────────────────────
    # Fresh fluid pooling reflects light — high V, low-medium S, irregular shape
    shiny_mask = (v > 180) & (s < 80)
    shiny_ratio = float(shiny_mask.mean())
    if shiny_ratio > 0.005:
        shiny_u8 = shiny_mask.astype(np.uint8) * 255
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(shiny_u8, connectivity=8)
        if num_labels > 1:
            largest = int(stats[1:, cv2.CC_STAT_AREA].max())
            total_px = img.shape[0] * img.shape[1]
            if largest / total_px > 0.005:
                confidence = min(0.65, 0.35 + shiny_ratio * 5)
                findings.append(
                    UnderbodyFinding(
                        label="possible_underbody_fluid_stain",
                        confidence=confidence,
                        details={"signal": "shiny_patch", "shiny_ratio": round(shiny_ratio, 3)},
                    )
                )

    return findings
