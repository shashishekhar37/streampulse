"""A minimal model registry: versioned model files plus metadata.

    outputs/registry/<model_name>/v0001.pkl
    outputs/registry/<model_name>/v0001.json   <- metrics, feature names, time, #samples
    outputs/registry/<model_name>/LATEST       <- version the server should load

The serving API (serve.py) always loads LATEST, so publishing a new version is how a
retrained model gets "deployed". Rolling back means pointing LATEST at an older version.
"""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path


class ModelRegistry:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def publish(self, name: str, model: object, metadata: dict) -> str:
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        version = f"v{len(list(d.glob('v*.pkl'))) + 1:04d}"
        with (d / f"{version}.pkl").open("wb") as f:
            pickle.dump(model, f)
        meta = {"name": name, "version": version,
                "published_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **metadata}
        (d / f"{version}.json").write_text(json.dumps(meta, indent=2, default=float))
        (d / "LATEST").write_text(version)
        return version

    def load(self, name: str, version: str | None = None) -> tuple[object, dict]:
        d = self.root / name
        version = version or (d / "LATEST").read_text().strip()
        with (d / f"{version}.pkl").open("rb") as f:
            model = pickle.load(f)
        return model, json.loads((d / f"{version}.json").read_text())

    def versions(self, name: str) -> list[dict]:
        return [json.loads(p.read_text()) for p in sorted((self.root / name).glob("v*.json"))]