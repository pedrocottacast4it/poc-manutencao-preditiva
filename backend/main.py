"""API da PoC de Manutenção Preditiva (WTTE-RNN) — HTTP stateless, compatível com AWS App Runner.

Na subida, o modelo roda uma vez sobre todo o conjunto de teste do C-MAPSS FD001 e as
predições (α, β por ciclo) ficam em memória; os endpoints só recortam e agregam.
"""

import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from backend.wtte import (
    FEATURES,
    SENSOR_LABELS,
    WtteModel,
    feature_matrix,
    load_cmapss,
    weibull_cdf,
    weibull_median,
    weibull_pdf,
    weibull_quantile,
)

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = Path(os.environ.get("FRONTEND_DIR", ROOT / "frontend"))
DATA = ROOT / "data" / "cmapss"
MODEL_DIR = Path(__file__).resolve().parent / "model"

RUL_CAP = 125

# Os 100 motores do C-MAPSS são usados como proxy de degradação de ativos rotativos;
# os nomes de equipamento abaixo são ilustrativos, para contextualizar a demo na usina.
ASSET_TYPES = [
    ("MOE", "Moenda"),
    ("EXC", "Exaustor de caldeira"),
    ("BFE", "Bomba de fermentação"),
    ("CEN", "Centrífuga de açúcar"),
    ("RED", "Redutor planetário"),
]
DISPLAY_SENSORS = ["s4", "s11", "s12", "s7", "s15", "s21"]

app = FastAPI(title="PoC Manutenção Preditiva — WTTE-RNN", version="0.3.0")


@lru_cache(maxsize=1)
def fleet_state():
    model = WtteModel.load(MODEL_DIR / "wtte_fd001.npz")
    metrics = json.loads((MODEL_DIR / "metrics.json").read_text())
    units = load_cmapss(DATA / "test_FD001.txt")
    true_rul = np.loadtxt(DATA / "RUL_FD001.txt")
    assets = []
    for unit, rows in units.items():
        feats = feature_matrix(rows)
        alpha, beta = model.predict_sequence(feats)
        prefix, kind = ASSET_TYPES[(unit - 1) % len(ASSET_TYPES)]
        assets.append(
            {
                "id": unit,
                "tag": f"{prefix}-{(unit - 1) // len(ASSET_TYPES) + 1:02d}",
                "type": kind,
                "feats": feats,
                "alpha": alpha,
                "beta": beta,
                "true_rul": float(true_rul[unit - 1]),
            }
        )
    return model, metrics, assets


def get_asset(asset_id: int):
    for asset in fleet_state()[2]:
        if asset["id"] == asset_id:
            return asset
    raise HTTPException(status_code=404, detail="Ativo não encontrado")


def snapshot(asset, cycle: int, horizon: int, threshold: float, warn: float):
    """Estado do ativo observado até `cycle` (1-based): RUL previsto e risco no horizonte."""
    a, b = asset["alpha"][cycle - 1], asset["beta"][cycle - 1]
    prob = float(weibull_cdf(horizon, a, b))
    status = "critico" if prob >= threshold else "atencao" if prob >= warn else "normal"
    last = len(asset["feats"])
    return {
        "id": asset["id"],
        "tag": asset["tag"],
        "type": asset["type"],
        "cycle": cycle,
        "cycles_observed": last,
        "alpha": float(a),
        "beta": float(b),
        "rul_p50": float(weibull_median(a, b)),
        "rul_p10": float(weibull_quantile(a, b, 0.1)),
        "rul_p90": float(weibull_quantile(a, b, 0.9)),
        "prob_failure": prob,
        "status": status,
        # RUL real no ciclo escolhido: o dataset informa quantos ciclos o ativo ainda
        # operou após o fim da série de teste — usado só para validar a predição.
        "true_rul": asset["true_rul"] + (last - cycle),
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/info")
def api_info():
    _, metrics, assets = fleet_state()
    return {
        "service": "pdm-wtte-rnn",
        "dataset": metrics["dataset"],
        "assets": len(assets),
        "features": [{"key": f, "label": SENSOR_LABELS[f]} for f in FEATURES],
        "model": {
            "architecture": "LSTM(64) → Weibull(α, β)",
            "units_fit": metrics["units_fit"],
            "units_val": metrics["units_val"],
            "epochs_run": metrics["epochs_run"],
        },
    }


@app.get("/api/fleet")
def fleet(
    horizon: int = Query(30, ge=1, le=200),
    threshold: float = Query(0.5, gt=0, lt=1),
    warn: float = Query(0.2, gt=0, lt=1),
):
    _, _, assets = fleet_state()
    items = [snapshot(a, len(a["feats"]), horizon, threshold, warn) for a in assets]
    items.sort(key=lambda s: -s["prob_failure"])
    return {"horizon": horizon, "threshold": threshold, "warn": warn, "assets": items}


@app.get("/api/assets/{asset_id}")
def asset_detail(
    asset_id: int,
    cycle: int | None = Query(None, ge=1),
    horizon: int = Query(30, ge=1, le=200),
    threshold: float = Query(0.5, gt=0, lt=1),
    warn: float = Query(0.2, gt=0, lt=1),
):
    asset = get_asset(asset_id)
    last = len(asset["feats"])
    cycle = min(cycle or last, last)
    snap = snapshot(asset, cycle, horizon, threshold, warn)

    alpha, beta = asset["alpha"], asset["beta"]
    true_rul = asset["true_rul"] + (last - np.arange(1, last + 1))
    t_max = max(snap["rul_p90"] * 1.6, horizon * 1.5, 20)
    grid = np.linspace(0, t_max, 120)
    return {
        **snap,
        "sensors": [
            {"key": s, "label": SENSOR_LABELS[s], "values": asset["feats"][:, FEATURES.index(s)].round(4).tolist()}
            for s in DISPLAY_SENSORS
        ],
        "trajectory": {
            "cycles": list(range(1, last + 1)),
            "alpha": alpha.round(4).tolist(),
            "beta": beta.round(4).tolist(),
            "rul_p50": weibull_median(alpha, beta).round(2).tolist(),
            "rul_p10": weibull_quantile(alpha, beta, 0.1).round(2).tolist(),
            "rul_p90": weibull_quantile(alpha, beta, 0.9).round(2).tolist(),
            "prob_failure": weibull_cdf(horizon, alpha, beta).round(4).tolist(),
            "true_rul": true_rul.tolist(),
        },
        "curve": {
            "t": grid.round(2).tolist(),
            "pdf": weibull_pdf(np.maximum(grid, 1e-6), snap["alpha"], snap["beta"]).round(6).tolist(),
            "survival": (1 - weibull_cdf(grid, snap["alpha"], snap["beta"])).round(5).tolist(),
        },
    }


@app.get("/api/validation")
def validation(
    horizon: int = Query(30, ge=1, le=200),
    threshold: float = Query(0.5, gt=0, lt=1),
):
    """Compara o WTTE-RNN com a Weibull populacional (sem sensores) no conjunto de teste."""
    _, metrics, assets = fleet_state()
    pop_a, pop_b = metrics["baseline"]["alpha"], metrics["baseline"]["beta"]

    true = np.array([a["true_rul"] for a in assets])
    age = np.array([len(a["feats"]) for a in assets], dtype=float)
    alpha = np.array([a["alpha"][-1] for a in assets])
    beta = np.array([a["beta"][-1] for a in assets])

    def residual(p):
        return pop_a * ((age / pop_a) ** pop_b - np.log1p(-p)) ** (1 / pop_b) - age

    def summary(pred, lo, hi):
        err = pred - true
        capped = np.minimum(pred, RUL_CAP) - np.minimum(true, RUL_CAP)
        return {
            "rmse": float(np.sqrt(np.mean(err**2))),
            "rmse_cap125": float(np.sqrt(np.mean(capped**2))),
            "mae": float(np.mean(np.abs(err))),
            "coverage_p10_p90": float(np.mean((true >= lo) & (true <= hi))),
            "mean_interval_width": float(np.mean(hi - lo)),
        }

    pred = weibull_median(alpha, beta)
    prob = weibull_cdf(horizon, alpha, beta)
    alerts, fails = prob >= threshold, true <= horizon
    tp, fp = int(np.sum(alerts & fails)), int(np.sum(alerts & ~fails))
    fn, tn = int(np.sum(~alerts & fails)), int(np.sum(~alerts & ~fails))
    return {
        "dataset": metrics["dataset"],
        "units_test": len(assets),
        "wtte_rnn": summary(pred, weibull_quantile(alpha, beta, 0.1), weibull_quantile(alpha, beta, 0.9)),
        "weibull_populacional": summary(residual(0.5), residual(0.1), residual(0.9)),
        "alerts": {
            "horizon": horizon,
            "threshold": threshold,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
        },
        "scatter": [
            {"tag": a["tag"], "true": float(t), "pred": float(p)} for a, t, p in zip(assets, true, pred)
        ],
        "history": metrics["history"],
    }


app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
