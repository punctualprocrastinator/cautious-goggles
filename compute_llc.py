"""
LLC computation using devinterp v1.3.2 (last stable callback-based API).

Install: pip install "devinterp==1.3.2"

v1 high-level entry point:
    from devinterp.slt.sampler import estimate_learning_coeff_with_summary
    result = estimate_learning_coeff_with_summary(model, loader, ...)
    llc = result["llc/mean"]

SGLD kwargs for v1:
    sampling_method_kwargs={"lr": ..., "nbeta": n*beta}
    (nbeta = n * beta, not gamma, not localization)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score

# devinterp v1.3.2 imports
from devinterp.slt.sampler import estimate_learning_coeff_with_summary
from devinterp.optim.sgld import SGLD


# ---------------------------------------------------------------------------
# 1. Probe
# ---------------------------------------------------------------------------

class ToxinProbe(nn.Module):
    def __init__(self, embed_dim: int = 1280):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)   # (batch,) raw logit


# ---------------------------------------------------------------------------
# 2. Train probe
# ---------------------------------------------------------------------------

def train_probe(
    embeddings: np.ndarray,
    labels: np.ndarray,
    embed_dim: int = None,
    lr: float = 1e-2,
    epochs: int = 200,
    device: str = "cpu",
) -> tuple:
    if embed_dim is None:
        embed_dim = embeddings.shape[1]

    probe = ToxinProbe(embed_dim).to(device)
    X = torch.tensor(embeddings, dtype=torch.float32).to(device)
    y = torch.tensor(labels, dtype=torch.float32).to(device)
    crit = nn.BCEWithLogitsLoss()
    opt  = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)

    loader = DataLoader(TensorDataset(X, y), batch_size=256, shuffle=True)
    probe.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad(); crit(probe(xb), yb).backward(); opt.step()

    probe.eval()
    with torch.no_grad():
        loss_star = crit(probe(X), y).item()

    print(f"Training done. Loss at w*: {loss_star:.4f}")
    return probe, loss_star


# ---------------------------------------------------------------------------
# 3. LLC estimation — devinterp v1.3.2 API
# ---------------------------------------------------------------------------

def estimate_llc(
    probe: ToxinProbe,
    embeddings: np.ndarray,
    labels: np.ndarray,
    lr: float = 1e-5,
    beta: float = 0.1,          # inverse temperature scale; nbeta = n * beta
    num_chains: int = 5,
    num_draws: int = 200,
    num_burnin_steps: int = 50,
    batch_size: int = 64,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """
    LLC via devinterp v1.3.2 estimate_learning_coeff_with_summary().

    v1 SGLD kwargs:
      lr    — SGLD step size
      nbeta — inverse temperature = n * beta  (NOT gamma/localization)
    """
    torch.manual_seed(seed)

    n     = len(labels)
    nbeta = n * beta

    X = torch.tensor(embeddings, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.float32)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    crit = nn.BCEWithLogitsLoss()

    def loss_fn(model, batch):
        xb, yb = batch
        xb, yb = xb.to(device), yb.to(device)
        return crit(model(xb), yb)

    # v1 high-level function — handles SGLD + LLC internally
    result = estimate_learning_coeff_with_summary(
        model=probe,
        loader=loader,
        loss_fn=loss_fn,
        sampling_method=SGLD,
        sampling_method_kwargs={"lr": lr, "nbeta": nbeta},
        num_chains=num_chains,
        num_draws=num_draws,
        num_burnin_steps=num_burnin_steps,
        device=device,
        seed=seed,
    )

    # v1 result keys: "llc/mean", "llc/std", "loss/mean", etc.
    llc_mean = float(result["llc/mean"])
    llc_std  = float(result.get("llc/std", 0.0))

    print(f"LLC = {llc_mean:.4f} ± {llc_std:.4f}")
    print(f"d/2 reference (regular model) = {embeddings.shape[1] / 2:.1f}")

    return {
        "llc_mean":     llc_mean,
        "llc_std":      llc_std,
        "loss_at_w_star": result.get("loss/mean", None),
        "lr":           lr,
        "nbeta":        nbeta,
        "beta":         beta,
        "num_chains":   num_chains,
        "num_draws":    num_draws,
        "n":            n,
    }


# ---------------------------------------------------------------------------
# 4. WAIC (beta = 1/log(n) per Watanabe 2013)
# ---------------------------------------------------------------------------

def estimate_waic(
    probe: ToxinProbe,
    embeddings: np.ndarray,
    labels: np.ndarray,
    num_chains: int = 3,
    num_draws: int = 100,
    lr: float = 1e-5,
    device: str = "cpu",
) -> dict:
    n = len(labels)
    waic_beta = 1.0 / np.log(n)
    r = estimate_llc(probe, embeddings, labels,
                      lr=lr, beta=waic_beta,
                      num_chains=num_chains, num_draws=num_draws, device=device)
    crit = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        Xt = torch.tensor(embeddings, dtype=torch.float32).to(device)
        yt = torch.tensor(labels, dtype=torch.float32).to(device)
        loss_star = crit(probe(Xt), yt).item()
    wbic = r['llc_mean'] / waic_beta + n * loss_star
    return {"wbic": float(wbic), "waic_beta": waic_beta, "n": n}


# ---------------------------------------------------------------------------
# 5. Sensitivity grid
# ---------------------------------------------------------------------------

LLC_GRID = [
    # (lr,   beta,  num_draws, label)
    (1e-5, 0.10, 200, "default"),
    (1e-5, 0.01, 200, "low_beta"),
    (1e-6, 0.10, 200, "low_lr"),
    (1e-5, 0.10, 500, "more_draws"),
]

def run_sensitivity(probe, embeddings, labels, device="cpu"):
    results = {}
    for lr, beta, n_draws, label in LLC_GRID:
        print(f"\n[{label}]")
        r = estimate_llc(probe, embeddings, labels,
                          lr=lr, beta=beta, num_draws=n_draws, device=device)
        results[label] = r

    vals = [v["llc_mean"] for v in results.values()]
    spread = (max(vals) - min(vals)) / (np.mean(vals) + 1e-9)
    print(f"\nSpread: {spread:.1%}  {'✓ stable' if spread < 0.20 else '✗ unstable'}")
    return results


# ---------------------------------------------------------------------------
# 6. Validation
# ---------------------------------------------------------------------------

def validate_on_known_rlct(device: str = "cpu"):
    """
    True RLCT for logistic regression = d/2 (regular model).
    If estimate is within 30% of d/2, the v1 wiring is correct.
    """
    print("=" * 50)
    print("Validation: LLC on toy logistic regression")
    print("Expected LLC ≈ d/2 = 5.0  (d=10, regular model)")
    print("=" * 50)
    torch.manual_seed(0)

    d, n = 10, 500
    X = torch.randn(n, d)
    y = (X @ torch.randn(d) > 0).float()

    probe, loss_star = train_probe(X.numpy(), y.numpy(), embed_dim=d, device=device)

    r = estimate_llc(probe, X.numpy(), y.numpy(),
                      lr=1e-5, beta=0.1,
                      num_chains=3, num_draws=100, device=device)

    true_rlct = d / 2
    err = abs(r["llc_mean"] - true_rlct) / true_rlct
    print(f"\nTrue RLCT:     {true_rlct:.3f}")
    print(f"Estimated LLC: {r['llc_mean']:.3f} ± {r['llc_std']:.3f}")
    print(f"Relative err:  {err:.1%}")
    print("✓ Correct" if err < 0.30 else
          "✗ Off. Try: lower lr, higher num_draws, adjust beta.")
    return r


if __name__ == "__main__":
    validate_on_known_rlct()