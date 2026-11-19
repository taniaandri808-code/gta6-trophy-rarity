"""Forecasting utility: OLS on logit-transformed percentages with quantile outputs."""

import json
import math
import os
from pathlib import Path

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

INPUT_DIR = Path(os.getenv("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.getenv("MAKEFUN_OUTPUT_DIR", "/output"))

def clamp(x):
    return max(0.01, min(99.99, x))

def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1 - p))

def sigmoid(l):
    return 100.0 / (1.0 + math.exp(-l))

def round4(x):
    r = round(x * 10000) / 10000
    return int(r) if r == int(r) else r

def ols(xs, ys):
    n = len(xs)
    if n == 0:
        return (0.0, 0.0, 0.0)
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    sxy = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    sxx = sum((x - x_mean) ** 2 for x in xs)
    if sxx == 0:
        slope = 0.0
    else:
        slope = sxy / sxx
    intercept = y_mean - slope * x_mean
    if n < 3:
        resid_std = 0.0
    else:
        resid = [y - (slope * x + intercept) for x, y in zip(xs, ys)]
        resid_std = math.sqrt(sum(r * r for r in resid) / (n - 2))
    return (slope, intercept, resid_std)

def median(v):
    if not v:
        return 0.0
    s = sorted(v)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return (s[mid - 1] + s[mid]) / 2.0

# ---------------------------------------------------------------------------
# Utility helpers for historical epoch extraction and player projection.
# ---------------------------------------------------------------------------
import datetime

def epochs(history):
    """Return a list of (t_hours, players, rates) tuples.

    Parameters
    ----------
    history : list of dict
        Each dict must contain keys ``players_tracked``, ``rates`` and
        ``at`` (an ISO timestamp with trailing ``Z``).

    The function identifies maximal runs where the pair
    ``(players_tracked, rates)`` is identical.  For each run it keeps only the
    last record, converting the ``at`` timestamp to hours elapsed since
    ``2026‑11‑19T00:00:00Z``.
    """
    res = []
    last = None
    for rec in history:
        key = (rec.get("players_tracked"), rec.get("rates"))
        if key != last:
            if last is not None:
                # compute time for the previous run using the last record
                t_hours = (datetime.datetime.fromisoformat(last["at"].replace("Z", "+00:00"))
                           - datetime.datetime(2026, 11, 19, 0, 0, 0, tzinfo=datetime.timezone.utc)).total_seconds() / 3600
                res.append((t_hours, last["players_tracked"], last["rates"]))
            last = rec
    # handle final run
    if last is not None:
        t_hours = (datetime.datetime.fromisoformat(last["at"].replace("Z", "+00:00"))
                   - datetime.datetime(2026, 11, 19, 0, 0, 0, tzinfo=datetime.timezone.utc)).total_seconds() / 3600
        res.append((t_hours, last["players_tracked"], last["rates"]))
    return res

def project_players(epochs, t_target):
    """Least‑squares projection of players at ``t_target``.

    Parameters
    ----------
    epochs : list of (t_hours, players, rates)
    t_target : float
        Target time in hours.
    """
    xs = []
    ys = []
    for t, players, _ in epochs:
        if players > 0:
            xs.append(math.log(1 + t))
            ys.append(math.log(players))
    if len(xs) < 2:
        slope = 0.7
    else:
        slope, _, _ = ols(xs, ys)
    slope = max(0.0, min(1.6, slope))
    if not epochs:
        return 0
    t_last, players_last, _ = epochs[-1]
    factor = ((1 + t_target) / (1 + t_last)) ** slope
    projected = players_last * factor
    low, high = players_last, 25 * players_last
    return min(max(projected, low), high)
def earned(epochs):
    """Return list of lists E[i][j] = rates[j][i] * players_j / 100.

    E never decreases across epochs; when a rate is None the previous value
    is carried forward.
    """
    if not epochs:
        return []
    num_trophies = len(epochs[0][2]) if epochs[0][2] else 0
    E = [[0.0 for _ in epochs] for _ in range(num_trophies)]
    for j, (t, players, rates) in enumerate(epochs):
        for i in range(num_trophies):
            rate = rates[i] if i < len(rates) else None
            if rate is None:
                val = E[i][j-1] if j > 0 else 0.0
            else:
                val = rate * players / 100.0
            if j > 0 and val < E[i][j-1]:
                val = E[i][j-1]
            E[i][j] = val
    return E

def group_of(trophy):
    """Return trophy group: 'platinum', 'story', or 'visible'."""
    if trophy.get('tier') == 'platinum':
        return 'platinum'
    if trophy.get('secret'):
        return 'story'
    return 'visible'

def trophy_slopes(epochs, trophies):
    """Fit slope of ln(E) on ln(1+t) for each trophy.

    Returns (slopes, group_medians) where slopes is a list (None if <3 usable
    epochs) and group_medians maps group name to median of non-None slopes
    (default 0.8)."""
    E = earned(epochs)
    t_vals = [ep[0] for ep in epochs]
    slopes = []
    for i, trophy in enumerate(trophies):
        if i >= len(E):
            slopes.append(None)
            continue
        xs = []
        ys = []
        for j, e_val in enumerate(E[i]):
            if e_val > 0:
                xs.append(math.log(1 + t_vals[j]))
                ys.append(math.log(e_val))
        if len(xs) < 3:
            slopes.append(None)
        else:
            slope, _, _ = ols(xs, ys)
            slope = max(0.0, min(2.0, slope))
            slopes.append(slope)
    # group medians
    groups = {}
    for trophy, slope in zip(trophies, slopes):
        g = group_of(trophy)
        groups.setdefault(g, []).append(slope)
    group_medians = {}
    for g, vals in groups.items():
        non_none = [v for v in vals if v is not None]
        group_medians[g] = median(non_none) if non_none else 0.8
    return slopes, group_medians



def forecast_case(case, name):
    history = case.get("history", [])
    trophies = case.get("trophies", [])
    horizon_minutes = case.get("horizon_minutes", 0)
    eps = epochs(history)
    t_target = (datetime.datetime.fromisoformat(case["target"].replace("Z", "+00:00"))
                - datetime.datetime(2026, 11, 19, 0, 0, 0, tzinfo=datetime.timezone.utc)).total_seconds() / 3600.0
    t_last = eps[-1][0] if eps else 0.0
    P_target = project_players(eps, t_target)
    P_last = eps[-1][1] if eps else 0.0
    E = earned(eps)
    slopes, group_medians = trophy_slopes(eps, trophies)
    identical = False
    if eps:
        fp, fr = eps[0][1], eps[0][2]
        identical = all(ep[1] == fp and ep[2] == fr for ep in eps)
    rows = []
    for i, trophy in enumerate(trophies):
        E_i = E[i] if i < len(E) else []
        E_last = E_i[-1] if E_i else 0.0
        b = slopes[i] if i < len(slopes) and slopes[i] is not None else group_medians.get(group_of(trophy), 0.8)
        exponent = b * (1 + t_last) / (1 + t_target)
        E_target = max(E_last, E_last * ((1 + t_target) / (1 + t_last)) ** exponent)
        if P_target > 0:
            E_target = min(E_target, 0.999 * P_target)
        r_med = 100.0 * E_target / P_target if P_target > 0 else 0.01
        r_med = clamp(r_med)
        m = logit(r_med)
        if identical:
            last_rates = history[-1].get("rates", []) if history else []
            lr = last_rates[i] if i < len(last_rates) else None
            m = logit(clamp(lr if lr is not None else 0.01))
            s = 2.6
        else:
            t_vals = [ep[0] for ep in eps]
            xs = []
            ys = []
            for j, e_val in enumerate(E_i):
                if e_val > 0:
                    xs.append(math.log(1 + t_vals[j]))
                    ys.append(math.log(e_val))
            _, _, resid_std = ols(xs, ys) if len(xs) >= 3 else (0.0, 0.0, 0.0)
            sE = max(resid_std, 0.25) + 0.35 * max(0.0, math.log(E_target / E_last) if E_last > 0 else 0.0)
            sP = 0.30 + 0.25 * math.log(P_target / P_last) if P_last > 0 else 0.30
            risk = 0.40 * math.sqrt(horizon_minutes / 1440.0)
            s = math.sqrt(sE * sE + sP * sP + risk * risk)
        qs = [round4(sigmoid(m + z * s)) for z in Z]
        qs.sort()
        rows.append({"id": trophy.get("id", i), "q": qs})
    return rows


def main():
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for infile in INPUT_DIR.glob("*.in"):
        with open(infile, "r") as f:
            case = json.load(f)
        name = infile.stem
        rows = forecast_case(case, name)
        out = {"trophies": rows}
        outfile = OUTPUT_DIR / f"{name}.out"
        with open(outfile, "w") as f:
            f.write(json.dumps(out, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()