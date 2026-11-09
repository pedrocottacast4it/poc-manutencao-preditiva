"""Treina o WTTE-RNN no C-MAPSS FD001 e exporta os artefatos usados pela API.

Uso (na raiz do repositório):
    uv venv -p 3.12 ml/.venv && VIRTUAL_ENV=ml/.venv uv pip install -r ml/requirements.txt
    ml/.venv/bin/python ml/train.py

Saídas:
    backend/model/wtte_fd001.npz  pesos da LSTM + normalização (inferência em NumPy)
    backend/model/metrics.json    histórico de treino, baseline Weibull e métricas de teste
"""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.wtte import (  # noqa: E402
    FEATURES,
    WtteModel,
    feature_matrix,
    load_cmapss,
    weibull_cdf,
    weibull_median,
    weibull_quantile,
)

DATA = ROOT / "data" / "cmapss"
OUT = ROOT / "backend" / "model"

SEED = 42
HIDDEN = 64
MAX_BETA = 10.0
VAL_UNITS = 20
CENSORED_COPIES = 4  # truncamentos censurados por unidade de treino
MIN_TRUNC = 30
EPOCHS = 400
PATIENCE = 40
BATCH = 32
LR = 2e-3
EPS = 1e-9

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)


# ── Montagem das sequências ──


def run_to_failure(feats: np.ndarray):
    """Unidade observada até a falha: alvo = ciclos restantes, evento observado (u=1)."""
    n = len(feats)
    return feats, np.arange(n - 1, -1, -1, dtype=float), np.ones(n)


def censored(feats: np.ndarray, cut: int):
    """Unidade 'ainda operando' no ciclo `cut`: só sabemos que a falha é depois (u=0).

    É o caso real de uma usina — a maior parte dos ativos não falhou ainda — e é o
    que a log-verossimilhança censurada do WTTE permite aproveitar no treino.
    """
    return feats[:cut], np.arange(cut - 1, -1, -1, dtype=float), np.zeros(cut)


def pad(seqs):
    n, t_max, f = len(seqs), max(len(s[0]) for s in seqs), seqs[0][0].shape[1]
    x = np.zeros((n, t_max, f), dtype=np.float32)
    y = np.zeros((n, t_max), dtype=np.float32)
    u = np.zeros((n, t_max), dtype=np.float32)
    mask = np.zeros((n, t_max), dtype=np.float32)
    lengths = np.zeros(n, dtype=np.int64)
    for i, (xs, ys, us) in enumerate(seqs):
        k = len(xs)
        x[i, :k], y[i, :k], u[i, :k], mask[i, :k], lengths[i] = xs, ys, us, 1, k
    return [torch.from_numpy(a) for a in (x, y, u, mask, lengths)]


# ── Modelo ──


class Wtte(nn.Module):
    def __init__(self, n_features: int, init_alpha: float):
        super().__init__()
        self.lstm = nn.LSTM(n_features, HIDDEN, batch_first=True)
        self.out = nn.Linear(HIDDEN, 2)
        self.init_alpha = init_alpha
        with torch.no_grad():
            # Começa com β ≈ 1 (exponencial) e α ≈ média do TTE: chute neutro.
            self.out.bias.copy_(torch.tensor([0.0, -math.log(MAX_BETA - 1.0)]))

    def forward(self, x):
        h, _ = self.lstm(x)
        o = self.out(h)
        alpha = self.init_alpha * torch.exp(o[..., 0])
        beta = MAX_BETA * torch.sigmoid(o[..., 1])
        return alpha, beta


def discrete_weibull_loglik(y, u, alpha, beta):
    """Log-verossimilhança da Weibull discreta com censura (Martinsson, 2016).

    u=1: log P(Y = y)  = log(exp(Λ(y+1) − Λ(y)) − 1) − Λ(y+1)
    u=0: log P(Y > y)  = −Λ(y+1),   com Λ(t) = (t/α)^β
    """
    h0 = ((y + EPS) / alpha) ** beta
    h1 = ((y + 1.0) / alpha) ** beta
    return u * torch.log(torch.expm1(h1 - h0) + EPS) - h1


def masked_nll(model, batch):
    x, y, u, mask, _ = batch
    alpha, beta = model(x)
    ll = discrete_weibull_loglik(y, u, alpha, beta)
    return -(ll * mask).sum() / mask.sum()


# ── Baseline: Weibull populacional (sem sensores) ──


def fit_population_weibull(lifetimes: np.ndarray) -> tuple[float, float]:
    t = torch.tensor(lifetimes, dtype=torch.float64)
    log_a = torch.tensor(math.log(lifetimes.mean()), dtype=torch.float64, requires_grad=True)
    log_b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([log_a, log_b], lr=0.05)
    for _ in range(3000):
        a, b = log_a.exp(), log_b.exp()
        ll = torch.log(b / a) + (b - 1) * torch.log(t / a) - (t / a) ** b
        loss = -ll.mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(log_a.exp()), float(log_b.exp())


def baseline_residual_quantile(age, alpha, beta, p):
    """Vida residual condicional: r tal que P(T ≤ age + r | T > age) = p."""
    return alpha * ((age / alpha) ** beta - np.log1p(-p)) ** (1.0 / beta) - age


# ── Métricas ──


def nasa_score(pred, true):
    d = pred - true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


RUL_CAP = 125  # convenção da literatura do C-MAPSS (RUL linear por partes)


def regression_metrics(pred, lo, hi, true):
    err = pred - true
    capped = np.minimum(pred, RUL_CAP) - np.minimum(true, RUL_CAP)
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "rmse_cap125": float(np.sqrt(np.mean(capped**2))),
        "mae": float(np.mean(np.abs(err))),
        "nasa_score": nasa_score(pred, true),
        "coverage_p10_p90": float(np.mean((true >= lo) & (true <= hi))),
        "mean_interval_width": float(np.mean(hi - lo)),
    }


def main() -> None:
    t0 = time.time()
    train_units = load_cmapss(DATA / "train_FD001.txt")
    test_units = load_cmapss(DATA / "test_FD001.txt")
    test_rul = np.loadtxt(DATA / "RUL_FD001.txt")

    ids = np.array(sorted(train_units))
    rng.shuffle(ids)
    val_ids, fit_ids = sorted(ids[:VAL_UNITS]), sorted(ids[VAL_UNITS:])

    feats = {u: feature_matrix(rows) for u, rows in train_units.items()}
    stacked = np.concatenate([feats[u] for u in fit_ids])
    mean, std = stacked.mean(axis=0), stacked.std(axis=0) + 1e-8
    norm = {u: ((f - mean) / std).astype(np.float32) for u, f in feats.items()}

    fit_seqs = []
    for u in fit_ids:
        fit_seqs.append(run_to_failure(norm[u]))
        n = len(norm[u])
        for cut in rng.integers(MIN_TRUNC, n, size=CENSORED_COPIES):
            fit_seqs.append(censored(norm[u], int(cut)))
    val_full = pad([run_to_failure(norm[u]) for u in val_ids])

    # Validação no mesmo formato do teste: série interrompida antes da falha.
    val_cuts = {u: int(rng.integers(MIN_TRUNC, len(norm[u]) - 5)) for u in val_ids}

    mean_tte = np.mean([np.mean(s[1]) for s in fit_seqs if s[2][0] == 1])
    init_alpha = float(-1.0 / math.log(1.0 - 1.0 / (mean_tte + 1.0)))

    model = Wtte(len(FEATURES), init_alpha)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=15)

    history, best, best_state, waited = [], float("inf"), None, 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = rng.permutation(len(fit_seqs))
        total = 0.0
        for start in range(0, len(order), BATCH):
            batch = pad([fit_seqs[i] for i in order[start : start + BATCH]])
            loss = masked_nll(model, batch)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(batch[0])
        train_loss = total / len(fit_seqs)

        model.eval()
        with torch.no_grad():
            val_loss = float(masked_nll(model, val_full))
        sched.step(val_loss)
        history.append({"epoch": epoch, "train_nll": train_loss, "val_nll": val_loss})

        if val_loss < best - 1e-4:
            best, waited = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
        if epoch % 10 == 0:
            print(f"época {epoch:3d}  treino {train_loss:.4f}  validação {val_loss:.4f}")
        if waited >= PATIENCE:
            print(f"early stopping na época {epoch} (melhor validação {best:.4f})")
            break

    model.load_state_dict(best_state)
    model.eval()

    # ── Exporta para NumPy e confere que o forward bate com o PyTorch ──
    lstm = model.lstm
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT / "wtte_fd001.npz",
        w_ih=lstm.weight_ih_l0.detach().numpy(),
        w_hh=lstm.weight_hh_l0.detach().numpy(),
        bias=(lstm.bias_ih_l0 + lstm.bias_hh_l0).detach().numpy(),
        w_out=model.out.weight.detach().numpy(),
        b_out=model.out.bias.detach().numpy(),
        mean=mean,
        std=std,
        init_alpha=init_alpha,
        max_beta=MAX_BETA,
    )
    np_model = WtteModel.load(OUT / "wtte_fd001.npz")
    probe = feats[val_ids[0]]
    a_np, b_np = np_model.predict_sequence(probe)
    with torch.no_grad():
        a_t, b_t = model(torch.from_numpy(((probe - mean) / std).astype(np.float32))[None])
    assert np.allclose(a_np, a_t[0].numpy(), rtol=1e-3), "forward NumPy diverge do PyTorch"
    assert np.allclose(b_np, b_t[0].numpy(), rtol=1e-3), "forward NumPy diverge do PyTorch"

    # ── Avaliação ──
    pop_alpha, pop_beta = fit_population_weibull(
        np.array([len(feats[u]) for u in fit_ids], dtype=float)
    )

    def evaluate(series: dict[int, np.ndarray], truth: dict[int, float]):
        wt_pred, wt_lo, wt_hi, bl_pred, bl_lo, bl_hi, true, p30 = [], [], [], [], [], [], [], []
        for u, f in series.items():
            a, b = np_model.predict_sequence(f)
            a, b = a[-1], b[-1]
            wt_pred.append(weibull_median(a, b))
            wt_lo.append(weibull_quantile(a, b, 0.1))
            wt_hi.append(weibull_quantile(a, b, 0.9))
            p30.append(weibull_cdf(30, a, b))
            age = len(f)
            bl_pred.append(baseline_residual_quantile(age, pop_alpha, pop_beta, 0.5))
            bl_lo.append(baseline_residual_quantile(age, pop_alpha, pop_beta, 0.1))
            bl_hi.append(baseline_residual_quantile(age, pop_alpha, pop_beta, 0.9))
            true.append(truth[u])
        arr = lambda v: np.asarray(v, dtype=float)  # noqa: E731
        true = arr(true)
        alerts = arr(p30) >= 0.5
        fails = true <= 30
        return {
            "wtte_rnn": regression_metrics(arr(wt_pred), arr(wt_lo), arr(wt_hi), true),
            "weibull_populacional": regression_metrics(arr(bl_pred), arr(bl_lo), arr(bl_hi), true),
            "alerta_h30_p50": {
                "verdadeiros_positivos": int(np.sum(alerts & fails)),
                "falsos_positivos": int(np.sum(alerts & ~fails)),
                "falsos_negativos": int(np.sum(~alerts & fails)),
                "verdadeiros_negativos": int(np.sum(~alerts & ~fails)),
            },
        }

    val_series = {u: feats[u][: val_cuts[u]] for u in val_ids}
    val_truth = {u: float(len(feats[u]) - val_cuts[u]) for u in val_ids}
    test_series = {u: feature_matrix(rows) for u, rows in test_units.items()}
    test_truth = {u: float(test_rul[u - 1]) for u in test_units}

    metrics = {
        "dataset": "NASA C-MAPSS FD001",
        "units_fit": len(fit_ids),
        "units_val": len(val_ids),
        "units_test": len(test_units),
        "censored_copies_per_unit": CENSORED_COPIES,
        "hidden": HIDDEN,
        "epochs_run": len(history),
        "best_val_nll": best,
        "init_alpha": init_alpha,
        "baseline": {"alpha": pop_alpha, "beta": pop_beta},
        "validation": evaluate(val_series, val_truth),
        "test": evaluate(test_series, test_truth),
        "history": history,
        "train_seconds": round(time.time() - t0, 1),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False))

    print(json.dumps({k: metrics[k] for k in ("validation", "test", "baseline")}, indent=2))
    print(f"artefatos salvos em {OUT} ({metrics['train_seconds']}s)")


if __name__ == "__main__":
    main()
