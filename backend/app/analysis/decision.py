from __future__ import annotations

from dataclasses import dataclass

from app.analysis.audio_analysis import AudioFinding
from app.analysis.image_quality import ImageQuality


@dataclass
class Decision:
    recommendation: str  # go | no_go | inconclusive
    confidence: str  # high | medium | low
    top_reasons: list[dict]
    breakdown: dict
    education: list[dict]
    next_steps: list[dict]


def decide(
    *,
    photo_qualities: list[ImageQuality],
    audio_findings: list[AudioFinding],
    dashboard_findings: list[dict] | None = None,
    underbody_findings: list[dict] | None = None,
    driven_km: int | None = None,
) -> Decision:
    # Confidence mostly comes from data quality/completeness in the MVP
    low_quality_count = sum(0 if q.ok else 1 for q in photo_qualities)
    dashboard_findings = dashboard_findings or []
    underbody_findings = underbody_findings or []
    if low_quality_count >= 2:
        confidence = "low"
    elif low_quality_count == 1 or not dashboard_findings:
        confidence = "medium"
    else:
        confidence = "high"

    top_reasons: list[dict] = []
    education: list[dict] = []
    next_steps: list[dict] = []

    # Quality-driven inconclusive
    if low_quality_count >= 2:
        top_reasons.append(
            {
                "severity": "high",
                "title": "Some photos were too blurry or poorly lit to assess",
                "evidence": {"type": "quality", "count": low_quality_count},
            }
        )
        next_steps.append({"type": "retake", "text": "Retake at least 2 photos in brighter light and keep the phone steady."})

    # Audio-driven caution (very conservative in MVP)
    audio_risk = "low"
    for f in audio_findings:
        if f.label in {"rough_or_unstable"} and f.confidence >= 0.5:
            audio_risk = "medium"
            top_reasons.append(
                {
                    "severity": "medium",
                    "title": "Engine audio suggests an unstable/rough pattern",
                    "evidence": {"type": "audio", "label": f.label, "confidence": f.confidence},
                }
            )
            education.append({"id": "idle_quality", "title": "What a rough idle can indicate", "priority": "medium"})

    if driven_km is not None and driven_km >= 180_000 and audio_risk in {"low", "medium"}:
        top_reasons.append(
            {
                "severity": "medium",
                "title": "High driven KM means engine wear risk is naturally higher",
                "evidence": {"type": "vehicle_usage", "driven_km": driven_km},
            }
        )

    dashboard_risk = "low"
    if dashboard_findings:
        dashboard_risk = "high"
        top_reasons.append(
            {
                "severity": "high",
                "title": "Dashboard warning indicator may be present",
                "evidence": {"type": "dashboard", "findings": dashboard_findings},
            }
        )

    underbody_risk = "low"
    if underbody_findings:
        underbody_risk = "medium"
        top_reasons.append(
            {
                "severity": "medium",
                "title": "Underbody image shows possible fluid-stain pattern",
                "evidence": {"type": "underbody", "findings": underbody_findings},
            }
        )

    # Recommendation logic (MVP)
    if low_quality_count >= 2:
        recommendation = "inconclusive"
    elif dashboard_risk == "high":
        recommendation = "no_go"
        confidence = "medium"
        next_steps.append({"type": "inspection", "text": "Check warning lights with OBD scan before any purchase decision."})
    elif audio_risk == "medium":
        recommendation = "go"
        next_steps.append(
            {
                "type": "inspection",
                "text": "If you proceed, ask for an inspection focused on engine idle quality and scan for codes.",
            }
        )
    else:
        recommendation = "go"
        confidence = "medium"
        next_steps.append({"type": "inspection", "text": "A professional inspection is still recommended before buying."})

    breakdown = {
        "body_paint": {"risk": "unknown", "findings": []},
        "tires_suspension": {"risk": underbody_risk, "findings": underbody_findings},
        "interior_water": {"risk": "unknown", "findings": []},
        "dashboard": {"risk": dashboard_risk, "findings": dashboard_findings},
        "engine_sound": {
            "risk": audio_risk,
            "findings": [{"label": f.label, "confidence": f.confidence, "details": f.details} for f in audio_findings],
        },
    }

    if not education:
        education.append({"id": "photo_angles", "title": "Best photo angles for a used-car check", "priority": "low"})

    return Decision(
        recommendation=recommendation,
        confidence=confidence,
        top_reasons=top_reasons or [{"severity": "low", "title": "No clear red flags detected in this quick check", "evidence": {}}],
        breakdown=breakdown,
        education=education,
        next_steps=next_steps,
    )

