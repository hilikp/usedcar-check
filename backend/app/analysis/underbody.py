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
    Very conservative heuristic for possible fluid stains/leak-like dark patches.
    This is intentionally weak and should only be treated as a prompt to inspect.
    """
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return []

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    s = hsv[:, :, 1]

    # Dark + relatively low saturation regions can indicate wet/oily underbody patches.
    mask = (v < 55) & (s < 70)
    ratio = float(mask.mean())

    findings: list[UnderbodyFinding] = []
    if ratio > 0.18:
        confidence = min(0.75, 0.35 + ratio)
        findings.append(
            UnderbodyFinding(
                label="possible_underbody_fluid_stain",
                confidence=confidence,
                details={"dark_patch_ratio": ratio},
            )
        )
    return findings

