from __future__ import annotations
from pathlib import Path
import math, sys

# bonus?
BONUS_ENABLED = False

# chemins
CUR = Path(__file__).parent
SRC = CUR / "sources"
sys.path.append(str(CUR))
sys.path.append(str(SRC))

# data
DATA_CSV = CUR / "data" / "data.csv"
DATA_THETA = CUR / "data" / "theta.json"

# import custom
from csv_handler import Dataset, CSVFormatError
from theta_handler import Thetas, save_thetas, load_thetas
from utils.console import C, ok, info, warn, fail, title

# variantes
LR = 0.01
EPOCHS = 50000
LOGS_EVERY = 1000
TOLERANCE = 1e-6
ENABLED_TOL = True

# set min et max pour normalisation (+1 si egal sinon division par 0)
def fit_minmax(X: list[float]) -> tuple[float, float]:
    mn, mx = min(X), max(X)
    if mx == mn: mx = mn + 1.0
    return mn, mx

# normalise entre 0 et 1
def to_minmax(X: list[float], mn: float, mx: float) -> list[float]:
    rng = mx - mn
    return [(x - mn) / rng for x in X]

def step(X: list[float], y: list[float], t0: float, t1: float, lr: float) -> tuple[float, float]:
    m = len(X)
    s, sx = 0.0, 0.0
    for xi, yi in zip(X, y):
        e = (t0 + t1 * xi) - yi
        s += e; sx += e * xi
    return (t0 - lr * (s / m), t1 - lr * (sx / m))

def mse(X: list[float], y: list[float], t0: float, t1: float) -> float:
    m = len(X)
    return sum(((t0 + t1 * xi) - yi) ** 2 for xi, yi in zip(X, y)) / m

def main() -> int:
    title("Training")
    data_path = DATA_CSV
    try:
        ds = Dataset.from_csv(data_path)
        ok(f"CSV chargé: {data_path.name}")
    except (FileNotFoundError, CSVFormatError) as e:
        fail(f"{e}")
        return 1

    X, y = ds.as_arrays()
    x_min, x_max = fit_minmax(X)
    Xn = to_minmax(X, x_min, x_max)
    info(f"range [{int(x_min)},{int(x_max)}] → normalisé [0,1]")

    # params
    lr = LR
    epochs = EPOCHS
    log_every = LOGS_EVERY
    info(f"lr={lr} epochs={epochs} log_every={log_every}")
    print()

    t0, t1 = 0.0, 0.0
    last = mse(Xn, y, t0, t1)

    print(f"\033[90mStarting training...\033[0m")
    for e in range(1, epochs + 1):
        n0, n1 = step(Xn, y, t0, t1, lr)
        if not (math.isfinite(n0) and math.isfinite(n1)):
            warn("STOP non-fini → baisse le \'lr\' frero")
            break
        # update des thetas
        t0 = n0
        t1 = n1

        if e % log_every == 0 or e in (1, epochs):
            cur = mse(Xn, y, t0, t1)
            # facultatif, stop si convergeance
            if abs(last - cur) < TOLERANCE and ENABLED_TOL:
                warn(f"STOP convergé: mse < {TOLERANCE}")
                break
            # log + precision
            info(
                f"\033[38;2;75;0;130mtheta0=\033[0m{t0:.4f}\t|  "
                f"\033[38;2;72;61;139mtheta1=\033[0m{t1:.4f}\t|  "
                f"\033[38;2;123;104;238mmse=\033[0m{cur:.2f}\t|  "
                f"\033[38;2;106;90;205mrmse=\033[0m{math.sqrt(cur):.2f}\t|  "
                f"\033[38;2;148;0;211mR²=\033[0m{1 - cur / (sum((yi - sum(y) / len(y)) ** 2 for yi in y) / len(y)):.4f}  "
                f"- [{e}/{epochs}]"
            )
            if cur > last * 10 and last > 0:
                warn("STOP divergence → baisse le \'lr\' frero")
                break
            last = cur

    # sauvegarde dans json
    save_thetas(Thetas(theta0=t0, theta1=t1, x_min=x_min, x_max=x_max))

    title("Results")
    info(f"theta0: {t0}")
    info(f"theta1: {t1}")
    info(f"x_min: {x_min}")
    info(f"x_max: {x_max}")
    ok(f"saved → {DATA_THETA}\n")

    if BONUS_ENABLED:
        from sources.bonus.graph_handler import render_graph
        render_graph(data_csv=DATA_CSV, theta_json=DATA_THETA)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
