from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.schemas import AnalyzeOut, CheckCreateIn, CheckCreateOut, JobOut, ResultOut
from app.analysis.audio_analysis import get_audio_duration_seconds
from app.core.storage import local_path, public_url, save_upload
from app.db.models import Check, Job, JobStatus, Media, MediaKind, Result
from app.db.session import get_db
from app.workers.tasks import analyze_check

router = APIRouter(prefix="/v1")


@router.post("/checks", response_model=CheckCreateOut)
def create_check(payload: CheckCreateIn, db: Session = Depends(get_db)) -> CheckCreateOut:
    manufacturer = payload.make or payload.manufacturer
    chk = Check(
        nickname=payload.nickname,
        odometer=payload.odometer if payload.odometer is not None else payload.driven_km,
        driven_km=payload.driven_km if payload.driven_km is not None else payload.odometer,
        manufacturer=manufacturer,
        model_name=payload.model_name,
        year=payload.year,
        trim=payload.trim,
    )
    db.add(chk)
    db.commit()
    return CheckCreateOut(check_id=chk.id)


@router.post("/checks/{check_id}/photos")
def upload_photos(
    check_id: str,
    slot: str | None = None,
    files: list[UploadFile] = File(
        ...,
        description="You can select multiple images at once. In Windows file picker, hold Ctrl and left-click each image.",
    ),
    db: Session = Depends(get_db),
):
    chk = db.get(Check, check_id)
    if not chk:
        raise HTTPException(status_code=404, detail="check_not_found")
    if not (1 <= len(files) <= 10):
        raise HTTPException(status_code=400, detail="photos_count_must_be_1_to_10_per_request")

    out = []
    for f in files:
        rel, size = save_upload("photos", f)
        m = Media(
            check_id=chk.id,
            kind=MediaKind.photo,
            filename=rel,
            content_type=f.content_type,
            bytes=size,
            slot=slot,
        )
        db.add(m)
        db.flush()  # Assign media ID before returning response.
        out.append({"media_id": m.id, "url": public_url(rel), "slot": slot})

    db.commit()
    return {"uploaded": out}


@router.post("/checks/{check_id}/audio")
def upload_audio(
    check_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    chk = db.get(Check, check_id)
    if not chk:
        raise HTTPException(status_code=404, detail="check_not_found")

    rel, size = save_upload("audio", file)
    duration_seconds = get_audio_duration_seconds(local_path(rel))
    if duration_seconds < 10.0:
        try:
            os.remove(local_path(rel))
        except OSError:
            pass
        raise HTTPException(status_code=400, detail="audio_too_short_min_10_seconds")
    m = Media(
        check_id=chk.id,
        kind=MediaKind.audio,
        filename=rel,
        content_type=file.content_type,
        bytes=size,
        slot="engine_audio",
    )
    db.add(m)
    db.flush()
    db.commit()
    return {"media_id": m.id, "url": public_url(rel), "audio_duration_seconds": duration_seconds}


@router.post("/checks/{check_id}/analyze", response_model=AnalyzeOut)
def analyze(check_id: str, db: Session = Depends(get_db)) -> AnalyzeOut:
    chk = db.get(Check, check_id)
    if not chk:
        raise HTTPException(status_code=404, detail="check_not_found")

    # Minimal completeness gate: at least 4 photos + 1 audio
    photos = [m for m in chk.media if m.kind == MediaKind.photo]
    audio = [m for m in chk.media if m.kind == MediaKind.audio]
    if len(photos) < 4 or len(audio) < 1:
        raise HTTPException(status_code=400, detail="need_at_least_4_photos_and_audio")

    job = Job(check_id=chk.id, status=JobStatus.queued, progress=0)
    db.add(job)
    db.commit()

    analyze_check.delay(job.id)
    return AnalyzeOut(job_id=job.id)


@router.get("/jobs/{job_id}", response_model=JobOut)
def get_job(job_id: str, db: Session = Depends(get_db)) -> JobOut:
    job = db.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="job_not_found")
    return JobOut(status=job.status.value, progress=job.progress / 100.0, error=job.error)


@router.get("/checks/{check_id}/result", response_model=ResultOut)
def get_result(check_id: str, db: Session = Depends(get_db)) -> ResultOut:
    res = db.get(Result, check_id)
    if not res:
        raise HTTPException(status_code=404, detail="result_not_found")
    return ResultOut(**res.payload)


@router.get("/reference/makes")
def get_reference_makes():
    # Starter list for UI dropdown. Can be expanded or replaced from DB/config later.
    return {
        "makes": [
            "Acura", "Alfa Romeo", "Audi", "BMW", "Buick", "BYD", "Cadillac", "Chevrolet", "Chrysler",
            "Citroen", "Dacia", "Daihatsu", "Dodge", "DS", "Ferrari", "Fiat", "Ford", "Geely", "Genesis",
            "GMC", "Great Wall", "Honda", "Hyundai", "Infiniti", "Isuzu", "Jaguar", "Jeep", "Kia",
            "Lamborghini", "Land Rover", "Lexus", "Maserati", "Mazda", "Mercedes-Benz", "MG", "Mini",
            "Mitsubishi", "Nissan", "Opel", "Peugeot", "Polestar", "Porsche", "Ram", "Renault",
            "Rolls-Royce", "SEAT", "Skoda", "Subaru", "Suzuki", "Tesla", "Toyota", "Volkswagen", "Volvo",
        ]
    }


@router.get("/reference/years")
def get_reference_years():
    return {"years": list(range(1990, 2027))}


@router.get("/media/{rel_path:path}")
def get_media(rel_path: str):
    # Basic traversal protection
    if ".." in rel_path or rel_path.startswith(("/", "\\")):
        raise HTTPException(status_code=400, detail="invalid_path")
    full = Path(local_path(rel_path))
    if not full.exists():
        raise HTTPException(status_code=404, detail="not_found")
    return FileResponse(str(full))

