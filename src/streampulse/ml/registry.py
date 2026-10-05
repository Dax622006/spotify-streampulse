# spotify-streampulse/src/streampulse/ml/registry.py
"""
Model registry — tracks and persists all 6 ML models and metadata.
R-ML-08: Model artifacts saved to artifacts/models/ with metadata.
R-ML-09: No hard-coded metrics; loaded from registry metadata.
"""
from __future__ import annotations
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib

from streampulse.config import MODEL_REGISTRY_FILE, MODELS_DIR
from streampulse.logging_setup import get_logger, log_event

logger = get_logger(__name__)


@dataclass
class ModelMetadata:
    model_id: str
    model_name: str
    model_type: str            # regression / classification / clustering / association
    features: list[str]
    target: str | None
    metrics: dict[str, Any]
    params: dict[str, Any]
    source_sha256: str
    trained_at: str
    artifact_path: str


class ModelRegistry:
    """Registry managing model persistence, metadata, and status."""

    def __init__(self, registry_file: Path = MODEL_REGISTRY_FILE, models_dir: Path = MODELS_DIR):
        self.registry_file = registry_file
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._entries: dict[str, ModelMetadata] = {}
        self.load()

    def load(self) -> None:
        """Load registry from JSON file if present."""
        if not self.registry_file.exists():
            self._entries = {}
            return
        try:
            data = json.loads(self.registry_file.read_text(encoding="utf-8"))
            self._entries = {k: ModelMetadata(**v) for k, v in data.items()}
        except Exception as exc:
            logger.error(f"Failed to load model registry: {exc}")
            self._entries = {}

    def save(self) -> None:
        """Persist registry to JSON file."""
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        data = {k: asdict(v) for k, v in self._entries.items()}
        self.registry_file.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def register(
        self,
        model_id: str,
        model_name: str,
        model_type: str,
        model_obj: Any,
        features: list[str],
        target: str | None,
        metrics: dict[str, Any],
        params: dict[str, Any],
        source_sha256: str,
    ) -> ModelMetadata:
        """Save model artifact and record metadata."""
        artifact_filename = f"{model_id}.joblib"
        artifact_path = self.models_dir / artifact_filename
        joblib.dump(model_obj, artifact_path)

        meta = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            model_type=model_type,
            features=features,
            target=target,
            metrics=metrics,
            params=params,
            source_sha256=source_sha256,
            trained_at=datetime.now(timezone.utc).isoformat(),
            artifact_path=str(artifact_path),
        )
        self._entries[model_id] = meta
        self.save()

        log_event(
            logger,
            logging.INFO,
            "MODEL_REGISTERED",
            f"Registered model '{model_id}' ({model_type})",
            model_id=model_id,
            metrics=metrics,
        )
        return meta

    def get_metadata(self, model_id: str) -> ModelMetadata | None:
        return self._entries.get(model_id)

    def load_model(self, model_id: str) -> Any | None:
        """Load pickled model artifact."""
        meta = self.get_metadata(model_id)
        if not meta:
            return None
        # Resolve by filename inside models_dir so the registry still works
        # when the project folder is moved (artifact_path may hold an old absolute path).
        artifact = self.models_dir / Path(str(meta.artifact_path).replace("\\", "/")).name
        if not artifact.exists():
            return None
        return joblib.load(artifact)

    def list_models(self) -> list[ModelMetadata]:
        return list(self._entries.values())

    def is_stale(self, model_id: str, current_sha: str) -> bool:
        """Check if model was trained on an older source file checksum."""
        meta = self.get_metadata(model_id)
        if not meta:
            return True
        return meta.source_sha256 != current_sha


# Singleton registry instance
registry = ModelRegistry()
