from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class AudioFinding:
    label: str  # e.g. normal, rough_idle, squeal_like, ticking_like, knock_like, unknown
    confidence: float  # 0..1
    details: dict


@dataclass
class AudioAnalysis:
    duration_seconds: float
    findings: list[AudioFinding]


def get_audio_duration_seconds(path: str) -> float:
    # Fast duration check used during upload validation.
    return float(librosa.get_duration(path=path))


def analyze_engine_audio(path: str) -> AudioAnalysis:
    """
    MVP-grade audio analysis: conservative heuristics based on spectral features.
    This is not a diagnosis; it only provides coarse risk signals.
    """
    y, sr = librosa.load(path, sr=22050, mono=True)
    duration_seconds = float(y.size / float(sr)) if sr else 0.0
    if y.size < sr * 2:
        return AudioAnalysis(
            duration_seconds=duration_seconds,
            findings=[AudioFinding(label="unknown", confidence=0.2, details={"reason": "audio_too_short"})],
        )

    # Basic features
    rms = librosa.feature.rms(y=y)[0]
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]

    rms_mean = float(np.mean(rms))
    rms_var = float(np.var(rms))
    zcr_mean = float(np.mean(zcr))
    centroid_mean = float(np.mean(centroid))

    findings: list[AudioFinding] = []

    # Rough/unstable idle proxy: high RMS variance relative to mean
    if rms_mean > 0.01 and (rms_var / (rms_mean**2 + 1e-9)) > 0.8:
        findings.append(
            AudioFinding(
                label="rough_or_unstable",
                confidence=0.55,
                details={"rms_mean": rms_mean, "rms_var": rms_var},
            )
        )

    # High-frequency / squeal-ish proxy: high spectral centroid + elevated ZCR
    if centroid_mean > 3500 and zcr_mean > 0.10:
        findings.append(
            AudioFinding(
                label="high_frequency_squeal_like",
                confidence=0.45,
                details={"centroid_mean": centroid_mean, "zcr_mean": zcr_mean},
            )
        )

    # Ticking-ish proxy: moderately high ZCR but not extremely high centroid
    if zcr_mean > 0.12 and centroid_mean < 4500:
        findings.append(
            AudioFinding(
                label="ticking_like",
                confidence=0.4,
                details={"centroid_mean": centroid_mean, "zcr_mean": zcr_mean},
            )
        )

    if not findings:
        findings.append(
            AudioFinding(
                label="no_clear_anomaly_detected",
                confidence=0.6,
                details={"rms_mean": rms_mean, "zcr_mean": zcr_mean, "centroid_mean": centroid_mean},
            )
        )

    return AudioAnalysis(duration_seconds=duration_seconds, findings=findings)

