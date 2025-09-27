#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
import sys, math

CUR = Path(__file__).resolve()
SRC = CUR.parents[1]
PROJ = SRC.parent
DATA = PROJ / "data"
sys.path.append(str(SRC))
sys.path.append(str(PROJ))

from csv_handler import Dataset, CSVFormatError  # type: ignore
from theta_handler import load_thetas            # type: ignore
from utils.console import title, info, ok, warn, fail  # type: ignore

# on aime la redondance
def _norm(x: float, mn: float, mx: float) -> float:
    return 0.0 if mx == mn else (x - mn) / (mx - mn)

# redondance++
def _predict(km: float, t0: float, t1: float, x_min: float, x_max: float) -> float:  # ŷ
    return t0 + t1 * _norm(km, x_min, x_max)

# n points entre a et b
def _linspace(a: float, b: float, n: int = 120) -> list[float]:  # points
    if n <= 1: return [a, b]
    step = (b - a) / (n - 1)
    return [a + i * step for i in range(n)]

def _fmt_thousands(n: float) -> str:
    return f"{int(n):,}".replace(",", "'")

# style + titre
def _setup_style(fig, ax, title_str: str):
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mtick
    plt.style.use("dark_background")
    fig.set_facecolor("#0e1117")
    ax.set_facecolor("#0e1117")
    for s in ax.spines.values():
        s.set_color("#3b4252")
        s.set_linewidth(1.2)
    ax.grid(True, ls="--", lw=0.7, color="#2a2f3a", alpha=0.8)
    ax.tick_params(colors="#cbd5e1", labelsize=10)
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: _fmt_thousands(x)))
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda y, p: _fmt_thousands(y) + " €"))
    ax.set_title(title_str, fontsize=13, fontweight="bold", color="#e5e7eb")
    ax.set_xlabel("km", fontsize=11, color="#e5e7eb")
    ax.set_ylabel("price (€)", fontsize=11, color="#e5e7eb")
    try:
        fig.canvas.manager.set_window_title(f"ft_linear_regression")
    except Exception:
        pass  # headless

# call depuis train et predict pour le graph
def render_graph(
    *,
    data_csv: str | Path | None = None,
    theta_json: str | Path | None = None,
    predicted_km: float | None = None,
    show: bool = True,
    title_str: str = "Linear regression"
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        warn("Impossible d'afficher le graph > Matplotlib manquant"); return

    title("Graph")

    data_csv = Path(data_csv) if data_csv else (DATA / "data.csv")
    theta_json = Path(theta_json) if theta_json else (DATA / "theta.json")

    try:
        ds = Dataset.from_csv(data_csv)
        X, y = ds.as_arrays()
        ok(f"CSV: {len(X)} points chargés")
    except (FileNotFoundError, CSVFormatError) as e:
        fail(f"CSV err: {e}"); return

    t = load_thetas(theta_json)
    if not all(map(math.isfinite, (t.theta0, t.theta1, t.x_min, t.x_max))):
        fail("theta.json invalide"); return
    info(f"theta0={t.theta0:.4f} theta1={t.theta1:.4f}")

    xs = _linspace(min(X), max(X), n=200)
    ys = [_predict(x, t.theta0, t.theta1, t.x_min, t.x_max) for x in xs]

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    _setup_style(fig, ax, title_str)

    # span du modele
    ax.axvspan(t.x_min, t.x_max, color="#1f2937", alpha=0.25, lw=0)

    # data
    ax.scatter(X, y, s=28, alpha=0.9, label="data", color="#93c5fd", edgecolors="none")

    # ligne droite du model
    ax.plot(xs, ys, lw=6, alpha=0.22, color="#22d3ee")
    ax.plot(xs, ys, lw=2.2, color="#22d3ee", label="model")

    # point pour resultat
    if predicted_km is not None:
        yp = _predict(predicted_km, t.theta0, t.theta1, t.x_min, t.x_max)
        ax.scatter([predicted_km], [yp], s=90, c="#f43f5e", ec="#ffffff", lw=1.2, zorder=5, label="prediction")
        ax.annotate(f"{_fmt_thousands(yp)} €",
                    (predicted_km, yp),
                    textcoords="offset points",
                    xytext=(10, 10),
                    color="#e5e7eb")

    leg = ax.legend(frameon=True, fontsize=9)
    if leg:
        frame = leg.get_frame()
        frame.set_facecolor("#111827")
        frame.set_edgecolor("#404757")
        frame.set_alpha(0.9)

    fig.tight_layout()

    if show: plt.show()
    else: plt.close(fig)
    print("")
