#!/usr/bin/env python3
from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
SOURCES_DIR = PROJECT_DIR / "sources"
sys.path.append(str(SOURCES_DIR))
CSV_FILENAME = "data.csv"

from csv_handler import Dataset, CSVFormatError

class C:
	R = "\033[31m"
	G = "\033[32m"
	Y = "\033[33m"
	B = "\033[34m"
	M = "\033[35m"
	C = "\033[36m"
	D = "\033[0m"

def ok(msg: str) -> None:
	print(f"{C.G}✔ {msg}{C.D}")

def fail(msg: str) -> None:
	print(f"{C.R}✘ {msg}{C.D}")

def info(msg: str) -> None:
	print(f"{C.C}ℹ {msg}{C.D}")

def warn(msg: str) -> None:
	print(f"{C.Y}⚠ {msg}{C.D}")

def title(msg: str) -> None:
	print(f"\n{C.M}=== {msg} ==={C.D}")

def head_tail(vals: list[float], k: int = 3) -> str:
	if len(vals) <= 2 * k:
		return "[" + ", ".join(f"{v:.0f}" for v in vals) + "]"
	return "[" + ", ".join(f"{v:.0f}" for v in vals[:k]) + " ... " + ", ".join(f"{v:.0f}" for v in vals[-k:]) + "]"

def render_table(km: list[float], price: list[float], rows: int = 8) -> None:
	rows = min(rows, len(km))
	w_idx = len(str(rows - 1))
	print(f"{C.B}{'i'.rjust(w_idx)}  {'km'.rjust(10)}  {'price'.rjust(10)}{C.D}")
	for i in range(rows):
		print(f"{str(i).rjust(w_idx)}  {km[i]:10.0f}  {price[i]:10.0f}")
	if len(km) > rows:
		print(" " * (w_idx + 1) + "  ...         ...")

def check(predicate: bool, msg_ok: str, msg_fail: str, failures: list[str]) -> None:
	if predicate:
		ok(msg_ok)
	else:
		fail(msg_fail)
		failures.append(msg_fail)

def main() -> int:
	data_path = (PROJECT_DIR / "data" / CSV_FILENAME) if len(sys.argv) < 2 else Path(sys.argv[1]).resolve()
	title("Chargement du dataset")
	info(f"Fichier: {data_path}")

	try:
		ds = Dataset.from_csv(data_path)
		ok("Lecture CSV réussie (entête valide, lignes parsées).")
	except FileNotFoundError as e:
		fail(f"Fichier introuvable: {e}")
		return 1
	except CSVFormatError as e:
		fail(f"Format CSV invalide: {e}")
		return 1
	except Exception as e:
		fail(f"Exception inattendue: {e}")
		return 1

	X = ds.features()
	y = ds.targets()

	title("Aperçu & statistiques")
	info(f"Nombre d'échantillons: {len(X)}")
	print(f"{C.B}km    {C.D}{head_tail(X)}")
	print(f"{C.B}price {C.D}{head_tail(y)}\n")
	render_table(X, y, rows=8)

	km_min, km_max = min(X), max(X)
	pr_min, pr_max = min(y), max(y)
	km_mean = sum(X) / len(X)
	pr_mean = sum(y) / len(y)
	print(f"\n{C.B}Stats km    {C.D}min={km_min:.2f}  max={km_max:.2f}  mean={km_mean:.2f}")
	print(f"{C.B}Stats price {C.D}min={pr_min:.2f}  max={pr_max:.2f}  mean={pr_mean:.2f}")

	title("Vérifications d'intégrité")
	failures: list[str] = []

	check(isinstance(X, list) and isinstance(y, list),
		  "features()/targets() renvoient des listes.",
		  "features()/targets() ne renvoient pas des listes.",
		  failures)

	check(len(X) == len(y) and len(X) > 0,
		  "X et y ont la même taille (>0).",
		  "Taille incohérente entre X et y, ou vide.",
		  failures)

	all_finite = all(math.isfinite(v) for v in X + y)
	check(all_finite,
		  "Toutes les valeurs sont finies (pas de inf/NaN).",
		  "Présence de valeurs non finies (inf/NaN).",
		  failures)

	non_negative = all(v >= 0 for v in X) and all(v >= 0 for v in y)
	check(non_negative,
		  "Toutes les valeurs sont non négatives.",
		  "Valeurs négatives détectées dans km/price.",
		  failures)

	indices = {0, len(X) // 2, len(X) - 1} if len(X) >= 3 else set(range(len(X)))
	try:
		for i in indices:
			_ = (X[i], y[i])
		ok("Alignement X[i] ↔ y[i] vérifié sur un échantillon de points.")
	except Exception:
		fail("Alignement X[i] ↔ y[i] invalide (indexation).")
		failures.append("Alignement X[i] ↔ y[i] invalide")

	km_reasonable = (km_min >= 0) and (km_max <= 1_000_000)
	check(km_reasonable,
		  "Plage km raisonnable [0 .. 1e6].",
		  "Plage km anormale (hors [0 .. 1e6]).",
		  failures)

	price_reasonable = (pr_min >= 0) and (pr_max <= 1_000_000)
	check(price_reasonable,
		  "Plage price raisonnable [0 .. 1e6].",
		  "Plage price anormale (hors [0 .. 1e6]).",
		  failures)

	title("Résumé")
	if not failures:
		ok("Intégrité CSV : OK ✅")
		return 0
	else:
		warn("Intégrité CSV : problèmes détectés ❌")
		for m in failures:
			print(f" - {C.R}{m}{C.D}")
		return 1

if __name__ == "__main__":
	raise SystemExit(main())
