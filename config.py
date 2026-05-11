"""
config.py -- Configuration for the Self-Evolving COBOL Pipeline.

Centralised config with environment variable overrides.
"""

from __future__ import annotations

import os
import logging
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - project still works without .env support
    load_dotenv = None

if load_dotenv:
    load_dotenv()

logger = logging.getLogger(__name__)


def _env(primary: str, legacy: str, default: str) -> str:
    return os.getenv(primary, os.getenv(legacy, default))


class Config:
    """Configuration singleton with setup and validation."""

    def __init__(self):
        # Database
        self.DB_PATH = _env("COBOL_DB_PATH", "DB_PATH", "cobevolve_full_run.db")

        # Output
        self.OUTPUT_DIR = _env("COBOL_OUTPUT_DIR", "OUTPUT_DIR", "./full_run_output")
        self.TEMP_DIR = _env("COBOL_TEMP_DIR", "TEMP_DIR", "/tmp/cobol_pipeline")
        self.CHROMA_PATH = _env("COBOL_CHROMA_PATH", "CHROMA_PATH", "./chroma_db")

        # Translation
        self.TARGET_LANGUAGE = os.getenv("TARGET_LANGUAGE", "python")
        self.MAX_REPAIR_RETRIES = int(os.getenv("MAX_REPAIR_RETRIES", "3"))

        # Ollama
        self.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
        self.GNUCOBOL_PATH = os.getenv("GNUCOBOL_PATH", "/usr/bin/cobc")

        # Models
        self.MODEL_TRANSLATION = os.getenv("MODEL_TRANSLATION", "llama3.2")
        self.MODEL_ANALYSIS = os.getenv("MODEL_ANALYSIS", "llama3.2")
        self.MODEL_REPAIR = os.getenv("MODEL_REPAIR", "llama3.2")
        self.MODEL_TESTGEN = os.getenv("MODEL_TESTGEN", "llama3.2")

        # Validation
        self.VALIDATION_THRESHOLD = float(os.getenv("VALIDATION_THRESHOLD", "0.8"))

        # Internal state
        self._setup_done = False

    def setup(self):
        """One-time setup: ensure output dir exists, pull models if needed."""
        if self._setup_done:
            return self

        # Create output directory
        Path(self.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.TEMP_DIR).mkdir(parents=True, exist_ok=True)
        Path(self.CHROMA_PATH).mkdir(parents=True, exist_ok=True)

        try:
            import ollama

            client = ollama.Client(host=self.OLLAMA_BASE_URL)
            installed = {
                name
                for model in client.list().get("models", [])
                for name in (
                    model.get("name", ""),
                    model.get("model", ""),
                )
                if name
            }
            installed_aliases = installed | {
                name.removesuffix(":latest") for name in installed
            }
            required_models = {
                self.MODEL_TRANSLATION,
                self.MODEL_ANALYSIS,
                self.MODEL_REPAIR,
                self.MODEL_TESTGEN,
            }

            for model_name in required_models:
                if model_name in installed_aliases:
                    continue
                logger.info("Pulling Ollama model: %s", model_name)
                client.pull(model_name)
        except Exception as exc:
            logger.warning("Unable to verify/pull Ollama models: %s", exc)

        self._setup_done = True
        logger.info("Config setup complete")
        return self

    def verify_ollama(self) -> bool:
        """Check if Ollama server is running and models are available."""
        try:
            import ollama
            client = ollama.Client(host=self.OLLAMA_BASE_URL)
            models = {
                name
                for model in client.list().get("models", [])
                for name in (
                    model.get("name", ""),
                    model.get("model", ""),
                )
                if name
            }
            model_aliases = models | {
                name.removesuffix(":latest") for name in models
            }
            required = {
                self.MODEL_TRANSLATION,
                self.MODEL_ANALYSIS,
                self.MODEL_REPAIR,
                self.MODEL_TESTGEN,
            }
            return required.issubset(model_aliases)
        except Exception:
            return False


# Global instance
config = Config()
