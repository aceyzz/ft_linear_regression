from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math

# toujours voulu tester OOP en python hehe

class CSVFormatError(Exception):
    """erreur parsing csv"""

@dataclass(frozen=True, slots=True)
class CarSample:
    km: float
    price: float

    def __post_init__(self) -> None:
        if math.isnan(self.km) or math.isnan(self.price):
            raise ValueError("NaN détecté.")
        if self.km < 0:
            raise ValueError(f"Kilométrage négatif: {self.km}")
        if self.price < 0:
            raise ValueError(f"Prix négatif: {self.price}")

class Dataset:
    """dataset immuable"""
    __slots__ = ("_samples",)

    def __init__(self, samples: list[CarSample]) -> None:
        if not samples:
            raise ValueError("Dataset vide.")
        object.__setattr__(self, "_samples", tuple(samples))

    @classmethod
    def from_csv(cls, path: str | Path, delimiter: str = ",") -> Dataset:
        """charge dataset csv avec km,price en header"""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Fichier introuvable: {p}")

        with p.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f, delimiter=delimiter)

            try:
                header = next(reader)
            except StopIteration:
                raise CSVFormatError("CSV vide: entête manquante.")

            if [h.strip().lower() for h in header] != ["km", "price"]:
                raise CSVFormatError(f"Entête invalide: {header!r}. Attendu: ['km','price'].")

            samples: list[CarSample] = []
            for i, row in enumerate(reader, start=2):
                if not row or all(not c.strip() for c in row):
                    continue  # skip lignes vides
                if len(row) < 2:
                    raise CSVFormatError(f"Ligne {i}: attendu 2 colonnes, trouvé {len(row)}.")
                try:
                    km = float(row[0].strip())
                    price = float(row[1].strip())
                    samples.append(CarSample(km, price))
                except ValueError as e:
                    raise CSVFormatError(f"Ligne {i}: valeurs invalides ({row!r}).") from e

        return cls(samples)

    # getters
    def features(self) -> list[float]:
        return [s.km for s in self._samples]

    def targets(self) -> list[float]:
        return [s.price for s in self._samples]

    def as_arrays(self) -> tuple[list[float], list[float]]:
        return self.features(), self.targets()
