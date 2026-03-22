from __future__ import annotations

from sqlalchemy.orm import Session

from app.analysis.audio_analysis import analyze_engine_audio
from app.analysis.dashboard_hook import detect_dashboard_warnings
from app.analysis.decision import decide
from app.analysis.image_quality import assess_image_quality
from app.analysis.underbody import analyze_underbody_image
from app.core.storage import local_path
from app.db.models import Check, Job, JobStatus, MediaKind, Result
from app.db.session import SessionLocal
from app.workers.celery_app import celery


def _set_job(db: Session, job: Job, *, status: JobStatus | None = None, progress: int | None = None, error: str | None = None):
    if status is not None:
        job.status = status
    if progress is not None:
        job.progress = progress
    if error is not None:
        job.error = error
    db.add(job)
    db.commit()


@celery.task(name="app.workers.tasks.analyze_check")
def analyze_check(job_id: str) -> None:
    db = SessionLocal()
    try:
        job = db.get(Job, job_id)
        if not job:
            return
        chk = db.get(Check, job.check_id)
        if not chk:
            _set_job(db, job, status=JobStatus.failed, error="check_not_found")
            return

        _set_job(db, job, status=JobStatus.running, progress=5)

        photos = [m for m in chk.media if m.kind == MediaKind.photo]
        audio = next((m for m in chk.media if m.kind == MediaKind.audio), None)
        if len(photos) < 4 or audio is None:
            _set_job(db, job, status=JobStatus.failed, error="missing_required_media")
            return

        photo_qualities = []
        dashboard_paths: list[str] = []
        underbody_paths: list[str] = []
        for i, m in enumerate(photos):
            p = local_path(m.filename)
            photo_qualities.append(assess_image_quality(p))
            if (m.slot or "").lower() in {"dashboard", "dash"}:
                dashboard_paths.append(p)
            if (m.slot or "").lower() in {"underbody", "bottom"}:
                underbody_paths.append(p)
            _set_job(db, job, progress=5 + int((i + 1) / max(len(photos), 1) * 50))

        _set_job(db, job, progress=60)

        audio_analysis = analyze_engine_audio(local_path(audio.filename))
        dashboard_findings = [
            {"label": f.label, "confidence": f.confidence, "details": f.details}
            for f in detect_dashboard_warnings(dashboard_paths)
        ]
        underbody_findings = []
        for p in underbody_paths:
            underbody_findings.extend(
                {"label": f.label, "confidence": f.confidence, "details": f.details}
                for f in analyze_underbody_image(p)
            )
        _set_job(db, job, progress=85)

        decision = decide(
            photo_qualities=photo_qualities,
            audio_findings=audio_analysis.findings,
            dashboard_findings=dashboard_findings,
            underbody_findings=underbody_findings,
            driven_km=chk.driven_km,
        )

        payload = {
            "recommendation": decision.recommendation,
            "confidence": decision.confidence,
            "top_reasons": decision.top_reasons,
            "breakdown": decision.breakdown,
            "education": decision.education,
            "next_steps": decision.next_steps,
            "audio_duration_seconds": audio_analysis.duration_seconds,
            "car_details": {
                "manufacturer": chk.manufacturer,
                "model_name": chk.model_name,
                "year": chk.year,
                "trim": chk.trim,
                "driven_km": chk.driven_km,
            },
        }

        # Upsert result
        existing = db.get(Result, chk.id)
        if existing:
            existing.payload = payload
            db.add(existing)
        else:
            db.add(Result(check_id=chk.id, payload=payload))
        db.commit()

        _set_job(db, job, status=JobStatus.done, progress=100)
    except Exception as e:
        job = db.get(Job, job_id)
        if job:
            _set_job(db, job, status=JobStatus.failed, error=str(e))
    finally:
        db.close()

