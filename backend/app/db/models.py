import enum
from datetime import datetime
from uuid import uuid4

from sqlalchemy import JSON, DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class JobStatus(str, enum.Enum):
    queued = "queued"
    running = "running"
    done = "done"
    failed = "failed"


class MediaKind(str, enum.Enum):
    photo = "photo"
    audio = "audio"


class Check(Base):
    __tablename__ = "checks"

    # Keep within VARCHAR(36) by using uuid hex (32 chars) + 4 char prefix.
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: f"chk_{uuid4().hex}")
    nickname: Mapped[str | None] = mapped_column(String(120), nullable=True)
    odometer: Mapped[int | None] = mapped_column(Integer, nullable=True)
    driven_km: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Vehicle details provided by the user for clearer education + reporting.
    manufacturer: Mapped[str | None] = mapped_column(String(120), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(120), nullable=True)
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    trim: Mapped[str | None] = mapped_column(String(120), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    media: Mapped[list["Media"]] = relationship(back_populates="check", cascade="all, delete-orphan")
    jobs: Mapped[list["Job"]] = relationship(back_populates="check", cascade="all, delete-orphan")
    result: Mapped["Result | None"] = relationship(back_populates="check", cascade="all, delete-orphan")


class Media(Base):
    __tablename__ = "media"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: f"med_{uuid4().hex}")
    check_id: Mapped[str] = mapped_column(ForeignKey("checks.id", ondelete="CASCADE"))
    kind: Mapped[MediaKind] = mapped_column(Enum(MediaKind))
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str | None] = mapped_column(String(120), nullable=True)
    bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    slot: Mapped[str | None] = mapped_column(String(40), nullable=True)  # e.g. front, rear, dash, engine, tire...
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    check: Mapped["Check"] = relationship(back_populates="media")


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: f"job_{uuid4().hex}")
    check_id: Mapped[str] = mapped_column(ForeignKey("checks.id", ondelete="CASCADE"))
    status: Mapped[JobStatus] = mapped_column(Enum(JobStatus), default=JobStatus.queued)
    progress: Mapped[int] = mapped_column(Integer, default=0)  # 0..100
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    check: Mapped["Check"] = relationship(back_populates="jobs")


class Result(Base):
    __tablename__ = "results"

    check_id: Mapped[str] = mapped_column(ForeignKey("checks.id", ondelete="CASCADE"), primary_key=True)
    payload: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    check: Mapped["Check"] = relationship(back_populates="result")

