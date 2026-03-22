from sqlalchemy import text

from app.db.models import Base
from app.db.session import engine


def _column_exists(conn, table: str, column: str) -> bool:
    row = conn.execute(
        text(
            "SELECT 1 FROM information_schema.columns WHERE table_name = :t AND column_name = :c"
        ),
        {"t": table, "c": column},
    ).first()
    return row is not None


def _ensure_check_columns() -> None:
    # We run lightweight schema evolution for local MVP dev.
    # (So you don't need migrations just to add new fields.)
    wanted = [
        ("checks", "driven_km", "INTEGER"),
        ("checks", "manufacturer", "VARCHAR(120)"),
        ("checks", "model_name", "VARCHAR(120)"),
        ("checks", "year", "INTEGER"),
        ("checks", "trim", "VARCHAR(120)"),
    ]
    with engine.begin() as conn:
        for table, column, ddl in wanted:
            if not _column_exists(conn, table, column):
                conn.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {ddl}'))


def init_db() -> None:
    Base.metadata.create_all(bind=engine)
    _ensure_check_columns()

