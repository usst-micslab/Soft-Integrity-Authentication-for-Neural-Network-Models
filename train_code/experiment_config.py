"""Resource roots for the SIA experiments."""

import os
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("SIA_DATA_ROOT", PROJECT_ROOT / "data"))
MODEL_ROOT = Path(os.environ.get("SIA_MODEL_ROOT", PROJECT_ROOT / "model"))


def checkpoint(relative_path):
    return str(MODEL_ROOT / relative_path)


def load_model_state(prefix):
    path = Path(f"{prefix}.pth")
    if not path.is_file() and path.is_relative_to(MODEL_ROOT / "backdoor_model"):
        archived = MODEL_ROOT / "backdoor_model" / path.name
        if archived.is_file():
            path = archived
    return torch.load(path, map_location="cpu", weights_only=True)


def load_classifier(path, projection_head):
    # The archived classifiers were serialized by scripts running as __main__.
    allowed = [(projection_head, "__main__.ProjectionHead"), torch.nn.Sequential, torch.nn.Linear, torch.nn.ReLU]
    with torch.serialization.safe_globals(allowed):
        model = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(model, projection_head):
        raise TypeError(f"{path} must contain a ProjectionHead checkpoint")
    return model
