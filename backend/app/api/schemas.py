from pydantic import BaseModel, Field


class CheckCreateIn(BaseModel):
    nickname: str | None = Field(default=None, max_length=120)
    odometer: int | None = Field(default=None, ge=0, le=2_000_000)
    driven_km: int | None = Field(default=None, ge=0, le=2_000_000)
    make: str | None = Field(default=None, max_length=120)
    manufacturer: str | None = Field(default=None, max_length=120)
    model_name: str | None = Field(default=None, max_length=120)
    year: int | None = Field(default=None, ge=1990, le=2026)
    trim: str | None = Field(default=None, max_length=120)


class CheckCreateOut(BaseModel):
    check_id: str


class AnalyzeOut(BaseModel):
    job_id: str


class JobOut(BaseModel):
    status: str
    progress: float
    error: str | None = None


class ResultOut(BaseModel):
    recommendation: str
    confidence: str
    top_reasons: list[dict]
    breakdown: dict
    education: list[dict]
    next_steps: list[dict]
    audio_duration_seconds: float | None = None
    car_details: dict | None = None

