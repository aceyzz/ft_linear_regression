from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json

_DEFAULT = Path(__file__).resolve().parents[1] / "data" / "theta.json"

@dataclass(slots=True)
class Thetas:
    theta0: float = 0.0
    theta1: float = 0.0
    x_min: float = 0.0
    x_max: float = 1.0

def load_thetas(path: str | Path = _DEFAULT) -> Thetas:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return Thetas()
    try:
        with p.open("r", encoding="utf-8") as f:
            d = json.load(f)
        return Thetas(
            float(d.get("theta0", 0.0)),
            float(d.get("theta1", 0.0)),
            float(d.get("x_min", 0.0)),
            float(d.get("x_max", 1.0)),
        )
    except Exception:
        return Thetas()

def save_thetas(t: Thetas, path: str | Path = _DEFAULT) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(asdict(t), f, ensure_ascii=False, indent=2)
