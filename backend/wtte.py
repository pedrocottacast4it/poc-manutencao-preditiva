"""WTTE-RNN (Weibull Time-To-Event RNN) — dados C-MAPSS e inferência em NumPy.

O treino roda em PyTorch (ml/train.py); a imagem de runtime só precisa de NumPy:
os pesos da LSTM são exportados para .npz e o forward é refeito aqui.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Colunas do C-MAPSS: unidade, ciclo, 3 settings operacionais, 21 sensores.
COLUMNS = ["unit", "cycle", "op1", "op2", "op3"] + [f"s{i}" for i in range(1, 22)]

# No FD001 (condição operacional única) os sensores 1, 5, 6, 10, 16, 18 e 19
# são constantes; ficam só os 14 que carregam sinal de degradação.
FEATURES = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]

SENSOR_LABELS = {
    "s2": "Temperatura saída LPC (°R)",
    "s3": "Temperatura saída HPC (°R)",
    "s4": "Temperatura saída LPT (°R)",
    "s7": "Pressão saída HPC (psia)",
    "s8": "Rotação física do fan (rpm)",
    "s9": "Rotação física do core (rpm)",
    "s11": "Pressão estática saída HPC (psia)",
    "s12": "Razão combustível / Ps30",
    "s13": "Rotação corrigida do fan (rpm)",
    "s14": "Rotação corrigida do core (rpm)",
    "s15": "Bypass ratio",
    "s17": "Entalpia de sangria",
    "s20": "Sangria de refrigeração HPT",
    "s21": "Sangria de refrigeração LPT",
}


def load_cmapss(path: Path) -> dict[int, np.ndarray]:
    """Lê um arquivo train/test do C-MAPSS → {unidade: matriz (ciclos × 26)}."""
    raw = np.loadtxt(path)
    units: dict[int, np.ndarray] = {}
    for unit in np.unique(raw[:, 0]).astype(int):
        rows = raw[raw[:, 0] == unit]
        units[int(unit)] = rows[np.argsort(rows[:, 1])]
    return units


def feature_matrix(rows: np.ndarray) -> np.ndarray:
    idx = [COLUMNS.index(f) for f in FEATURES]
    return rows[:, idx]


# ── Distribuição de Weibull (parametrização α = escala, β = forma) ──


def weibull_quantile(alpha, beta, p):
    """Tempo t tal que P(T ≤ t) = p."""
    return alpha * (-np.log1p(-p)) ** (1.0 / beta)


def weibull_median(alpha, beta):
    return weibull_quantile(alpha, beta, 0.5)


def weibull_cdf(t, alpha, beta):
    return 1.0 - np.exp(-((np.asarray(t, dtype=float) / alpha) ** beta))


def weibull_pdf(t, alpha, beta):
    t = np.asarray(t, dtype=float)
    return (beta / alpha) * (t / alpha) ** (beta - 1) * np.exp(-((t / alpha) ** beta))


# ── Modelo exportado ──


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


@dataclass
class WtteModel:
    w_ih: np.ndarray  # (4H, F) — ordem dos gates do PyTorch: i, f, g, o
    w_hh: np.ndarray  # (4H, H)
    bias: np.ndarray  # (4H,) — b_ih + b_hh
    w_out: np.ndarray  # (2, H)
    b_out: np.ndarray  # (2,)
    mean: np.ndarray  # normalização z-score (estatísticas do treino)
    std: np.ndarray
    init_alpha: float
    max_beta: float

    @classmethod
    def load(cls, path: Path) -> "WtteModel":
        z = np.load(path)
        return cls(
            w_ih=z["w_ih"],
            w_hh=z["w_hh"],
            bias=z["bias"],
            w_out=z["w_out"],
            b_out=z["b_out"],
            mean=z["mean"],
            std=z["std"],
            init_alpha=float(z["init_alpha"]),
            max_beta=float(z["max_beta"]),
        )

    def predict_sequence(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Roda a LSTM ciclo a ciclo e devolve (α, β) previstos para cada ciclo.

        Como a rede é causal, a saída no ciclo t usa apenas o histórico até t —
        o mesmo que o modelo veria em produção, em modo shadow.
        """
        x = (features - self.mean) / self.std
        hidden = self.w_hh.shape[1]
        h = np.zeros(hidden)
        c = np.zeros(hidden)
        out = np.empty((len(x), 2))
        for t, xt in enumerate(x):
            gates = self.w_ih @ xt + self.w_hh @ h + self.bias
            i, f, g, o = np.split(gates, 4)
            c = _sigmoid(f) * c + _sigmoid(i) * np.tanh(g)
            h = _sigmoid(o) * np.tanh(c)
            out[t] = self.w_out @ h + self.b_out
        alpha = self.init_alpha * np.exp(out[:, 0])
        beta = self.max_beta * _sigmoid(out[:, 1])
        return alpha, beta
