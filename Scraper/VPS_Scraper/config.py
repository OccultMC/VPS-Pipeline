"""Configuration loaded from .env file."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).parent / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    # Try .env.example as fallback for first run
    _example = Path(__file__).parent / ".env.example"
    if _example.exists():
        load_dotenv(_example)


@dataclass
class Settings:
    # Cloudflare R2
    r2_endpoint_url: str = field(default_factory=lambda: os.getenv("R2_ENDPOINT_URL", ""))
    r2_access_key_id: str = field(default_factory=lambda: os.getenv("R2_ACCESS_KEY_ID", ""))
    r2_secret_access_key: str = field(default_factory=lambda: os.getenv("R2_SECRET_ACCESS_KEY", ""))
    r2_bucket_name: str = field(default_factory=lambda: os.getenv("R2_BUCKET_NAME", "streetview-data"))

    # Vast.AI
    vastai_api_key: str = field(default_factory=lambda: os.getenv("VASTAI_API_KEY", ""))

    # Docker images
    pipeline_docker_image: str = field(
        default_factory=lambda: os.getenv("PIPELINE_DOCKER_IMAGE", "ghcr.io/occultmc/vpr-pipeline:latest")
    )
    builder_docker_image: str = field(
        default_factory=lambda: os.getenv("BUILDER_DOCKER_IMAGE", "ghcr.io/occultmc/vpr-builder:latest")
    )

    def validate(self) -> list[str]:
        """Return list of missing required fields."""
        missing = []
        if not self.r2_endpoint_url:
            missing.append("R2_ENDPOINT_URL")
        if not self.r2_access_key_id:
            missing.append("R2_ACCESS_KEY_ID")
        if not self.r2_secret_access_key:
            missing.append("R2_SECRET_ACCESS_KEY")
        if not self.vastai_api_key:
            missing.append("VASTAI_API_KEY")
        return missing


settings = Settings()
