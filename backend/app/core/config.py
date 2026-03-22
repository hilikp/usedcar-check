from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=None, extra="ignore")

    app_env: str = "dev"

    database_url: str
    redis_url: str

    storage_backend: str = "local"  # local | s3 (later)
    local_storage_dir: str = "/data/uploads"
    public_base_url: str = "http://localhost:8000"


settings = Settings()

