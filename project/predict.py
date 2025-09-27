from __future__ import annotations
from pathlib import Path
import sys, math

# bonus?
BONUS_ENABLED = False

# chemins
CUR = Path(__file__).parent
SRC = CUR / "sources"
sys.path.append(str(CUR))
sys.path.append(str(SRC))

# data
DATA_THETA = CUR / "data" / "theta.json"

# import custom
from theta_handler import load_thetas
from utils.console import C, ok, info, fail, title

def normalize(x: float, mn: float, mx: float) -> float:
    if mx == mn:
        return 0.0
    return (x - mn) / (mx - mn)

def predict_price(km: float) -> float | None:
    t = load_thetas()
    if not math.isfinite(t.theta0) or not math.isfinite(t.theta1):
        return None
    km_norm = normalize(km, t.x_min, t.x_max)
    return t.theta0 + t.theta1 * km_norm

# si pas d'arg, interactif
def prompt_km() -> float | None:
    try:
        km_str = input("Entrez le kilométrage du véhicule: ")
        km = float(km_str)
        if km < 0:
            fail("Le kilométrage doit être >= 0.")
            return None
        return km
    except ValueError:
        fail("Argument invalide: km doit être un nombre >= 0.")
        return None

def main() -> int:
    title("Prediction")

    if len(sys.argv) < 2:
        km = prompt_km()
        if km is None:
            return 1
    else:
        try:
            km = float(sys.argv[1])
            if km < 0:
                raise ValueError("km négatif")
        except ValueError:
            fail("Argument invalide: km doit être un nombre >= 0\n")
            return 1

    price = predict_price(km)

    info(f"Kilométrage: {km:,.0f} km")
    info(f"Prix estimé: {price:,.2f} €")
    print("")

    if BONUS_ENABLED:
        from sources.bonus.graph_handler import render_graph
        render_graph(data_csv=CUR/"data/data.csv", theta_json=DATA_THETA, predicted_km=km)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
