"""Load repository-wide environment variables from one boundary."""

from pathlib import Path

from dotenv import load_dotenv


DEFAULT_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def load_repository_env(env_path: Path = DEFAULT_ENV_PATH) -> None:
    """Load shared runtime settings without overriding process variables."""
    load_dotenv(env_path)
