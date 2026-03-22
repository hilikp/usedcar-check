from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ImageQuality:
    ok: bool
    blur_score: float
    brightness: float
    issues: list[str]


def assess_image_quality(path: str) -> ImageQuality:
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return ImageQuality(ok=False, blur_score=0.0, brightness=0.0, issues=["unreadable_image"])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur heuristic: variance of Laplacian
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Brightness heuristic: mean pixel intensity
    brightness = float(gray.mean())

    issues: list[str] = []
    if blur_score < 60:  # conservative threshold; tune later
        issues.append("blurry")
    if brightness < 45:
        issues.append("too_dark")
    if brightness > 210:
        issues.append("too_bright")

    return ImageQuality(ok=len(issues) == 0, blur_score=blur_score, brightness=brightness, issues=issues)

