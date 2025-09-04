"""
Core model loading and inference utilities for Plant Doctor Streamlit app.

This module is intentionally UI-agnostic so it can be reused in CLI, Flask,
or batch pipelines. Streamlit-specific caching is handled in `app.py`.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as keras_image


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOGGER = logging.getLogger("plant_doctor")
if not LOGGER.handlers:
    _handler = logging.StreamHandler()
    _fmt = logging.Formatter("[%(levelname)s] %(asctime)s - %(name)s - %(message)s")
    _handler.setFormatter(_fmt)
    LOGGER.addHandler(_handler)
LOGGER.setLevel(logging.INFO)


# -----------------------------------------------------------------------------
# Exceptions
# -----------------------------------------------------------------------------
class PredictorError(Exception):
    """Base error for Predictor."""


class ModelNotLoadedError(PredictorError):
    """Raised when predict() is called before load()."""


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Config:
    """Configuration for model inference."""
    model_path: Path = Path("model/plantdisease.keras")
    labels_path: Path = Path("labels.json")
    # img_size is now optional; if None we infer from model.input_shape
    img_size: int | None = None


class Predictor:
    """Keras predictor wrapper with preprocessing and softmax output."""

    def __init__(self, config: Config | None = None) -> None:
        self.config = config or Config()
        self._model = None
        self._labels: List[str] = []
        self._metadata: Dict[str, Any] = {}
        self._target_size: Tuple[int, int] | None = None  # (H, W)
        self._input_channels: int | None = None

    @property
    def labels(self) -> List[str]:
        return self._labels

    @property
    def input_size(self) -> Tuple[int, int, int] | None:
        """Returns (H, W, C) if model is loaded, else None."""
        if self._target_size and self._input_channels:
            h, w = self._target_size
            return (h, w, self._input_channels)
        return None

    def load(self) -> None:
        """Load the model, label names, and optional metadata."""
        cfg = self.config
        if not cfg.model_path.exists():
            raise PredictorError(f"Model not found: {cfg.model_path.resolve()}")
        if not cfg.labels_path.exists():
            raise PredictorError(f"Labels not found: {cfg.labels_path.resolve()}")

        LOGGER.info("Loading labels from %s", cfg.labels_path)
        raw = json.loads(cfg.labels_path.read_text())
        # Accept either list[str] or list[dict{name,cause,cure}]
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            self._metadata = {
                item["name"]: {k: v for k, v in item.items() if k != "name"} for item in raw
            }
            self._labels = [item["name"] for item in raw]
        elif isinstance(raw, list):
            self._metadata = {}
            self._labels = list(raw)
        else:
            raise PredictorError(
                "labels.json must be a list of class names or objects with a 'name' field."
            )

        LOGGER.info("Loading model from %s", cfg.model_path)
        self._model = load_model(cfg.model_path)

        # Infer expected input size if not explicitly provided
        input_shape = getattr(self._model, "input_shape", None)
        # Handle multi-input models
        if isinstance(input_shape, (list, tuple)) and input_shape and isinstance(input_shape[0], (list, tuple)):
            input_shape = input_shape[0]

        if not input_shape or len(input_shape) != 4:
            raise PredictorError(f"Unsupported input shape: {input_shape}")

        _, H, W, C = input_shape

        if cfg.img_size is not None:
            # Allow explicit override but keep channels from model
            self._target_size = (cfg.img_size, cfg.img_size)
            self._input_channels = int(C)
        else:
            self._target_size = (int(H), int(W))
            self._input_channels = int(C)

        if self._input_channels != 3:
            raise PredictorError(f"Expected 3-channel RGB images, got C={self._input_channels}")

        LOGGER.info(
            "Model loaded successfully. %d classes detected. Expecting %dx%dx%d input.",
            len(self._labels), self._target_size[0], self._target_size[1], self._input_channels
        )

    def predict(self, image_path: Path) -> Dict[str, float | str | Dict[str, float] | Dict[str, Any]]:
        """Run prediction on a single image path.

        Args:
            image_path: Path to an RGB image.

        Returns:
            dict with top label, confidence (float), per-class probabilities, and optional info.
        """
        if self._model is None:
            raise ModelNotLoadedError("Model not loaded. Call load() first.")
        if self._target_size is None:
            raise PredictorError("Target input size is not set. Did load() succeed?")

        H, W = self._target_size
        img = keras_image.load_img(image_path, target_size=(H, W))
        x = keras_image.img_to_array(img).astype("float32")
        # Normalize as in training; your original code used /255.0
        x = np.expand_dims(x, axis=0) / 255.0  # (1, H, W, 3)

        preds = self._model.predict(x, verbose=0)[0]
        top_idx = int(np.argmax(preds))
        label = self._labels[top_idx] if self._labels else str(top_idx)
        info = self._metadata.get(label, {}) if hasattr(self, "_metadata") else {}
        result = {
            "label": label,
            "confidence": float(preds[top_idx]),
            "probs": {self._labels[i]: float(p) for i, p in enumerate(preds)} if self._labels else {},
            "info": info,
        }
        return result
