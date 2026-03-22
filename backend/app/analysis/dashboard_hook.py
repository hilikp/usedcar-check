from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DashboardFinding:
    label: str
    confidence: float
    details: dict


def detect_dashboard_warnings(image_paths: list[str]) -> list[DashboardFinding]:
    """
    Hook point for warning-light detection.
    MVP behavior: returns no detections (safe default) but keeps API shape stable.
    """
    _ = image_paths
    return []

