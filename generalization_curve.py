"""
Generalization curve experiment.

For each sequence identity bin (natural toxins at varying distances from training set),
compute:
  1. Probe AUROC
  2. BLAST sensitivity
  3. Probe cross-entropy loss
  4. WAIC (from SGLD)

The LLC is computed ONCE at the trained probe weights (it's a property of w*, not of
the test data). The generalization curve plots AUROC vs identity and overlays the LLC
as a reference — testing whether LLC predicts the identity threshold at which AUROC drops.
"""

import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
from compute_llc import ToxinProbe, EmbeddingDataset, estimate_waic

import torch.nn as nn


# ---------------------------------------------------------------------------
# Identity binning
# ---------------------------------------------------------------------------

IDENTITY_BINS = [
    (0.90, 1.00, "90–100%"),   # essentially training distribution
    (0.70, 0.90, "70–90%"),
    (0.50, 0.70, "50–70%"),
    (0.30, 0.50, "30–50%"),    # below typical BLAST screening threshold
    (0.10, 0.30, "10–30%"),    # deep out-of-distribution
    (0.00, 0.10, "<10%"),      # maximally novel
]


def assign_identity_bins(
    max_identities: np.ndarray,  # shape (N,), max identity of each test seq to training set
) -> list[np.ndarray]:
    """Returns list of boolean index arrays, one per bin."""
    return [
        (max_identities >= lo) & (max_identities < hi)
        for lo, hi, _ in IDENTITY_BINS
    ]


# ---------------------------------------------------------------------------
# Per-bin evaluation
# ---------------------------------------------------------------------------

def evaluate_probe_on_bin(
    probe: ToxinProbe,
    embeddings: np.ndarray,
    labels: np.ndarray,
    mask: np.ndarray,
    device: str = "cpu",
) -> dict:
    """
    Compute AUROC and cross-entropy loss on a subset of sequences
    defined by boolean mask.
    """
    if mask.sum() < 5:
        return {"auroc": float("nan"), "loss": float("nan"), "n": int(mask.sum())}

    X = torch.tensor(embeddings[mask], dtype=torch.float32).to(device)
    y = torch.tensor(labels[mask], dtype=torch.float32).to(device)

    probe.eval()
    with torch.no_grad():
        logits = probe(X)
        loss = nn.BCEWithLogitsLoss()(logits, y).item()
        probs = torch.sigmoid(logits).cpu().numpy()

    y_np = labels[mask]
    auroc = roc_auc_score(y_np, probs) if len(np.unique(y_np)) > 1 else float("nan")

    return {
        "auroc": auroc,
        "loss": loss,
        "n": int(mask.sum()),
        "n_positive": int(y_np.sum()),
    }


def evaluate_blast_on_bin(
    blast_identities: np.ndarray,   # shape (N,), BLAST top-hit identity for each test seq
    labels: np.ndarray,
    mask: np.ndarray,
    blast_threshold: float = 0.30,  # standard screening cutoff
) -> dict:
    """
    BLAST sensitivity = fraction of true positives (toxins) flagged by
    BLAST at the given identity threshold.

    BLAST flags a sequence if its max identity to any training toxin >= blast_threshold.
    blast_identities[i] = the max identity of test sequence i to any training sequence.
    """
    if mask.sum() < 5:
        return {"sensitivity": float("nan"), "specificity": float("nan"), "n": 0}

    y_bin = labels[mask]
    pred_bin = (blast_identities[mask] >= blast_threshold).astype(int)

    tp = ((pred_bin == 1) & (y_bin == 1)).sum()
    fn = ((pred_bin == 0) & (y_bin == 1)).sum()
    tn = ((pred_bin == 0) & (y_bin == 0)).sum()
    fp = ((pred_bin == 1) & (y_bin == 0)).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    return {
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": int(tp), "fn": int(fn), "tn": int(tn), "fp": int(fp),
        "n": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# Main generalization curve
# ---------------------------------------------------------------------------

def run_generalization_curve(
    probe: ToxinProbe,
    test_embeddings: np.ndarray,   # all test sequences (natural + redesigned)
    test_labels: np.ndarray,       # 1 = toxin, 0 = non-toxic
    max_identities: np.ndarray,    # max sequence identity to training set, per test seq
    blast_identities: np.ndarray,  # BLAST top-hit identity to training toxins, per test seq
    llc_value: float,              # LLC computed at trained probe weights
    llc_std: float,
    train_embeddings: np.ndarray,  # needed for per-bin WAIC
    train_labels: np.ndarray,
    device: str = "cpu",
    seed: int = 42,
) -> dict:
    """
    Run the full generalization curve experiment.
    Returns dict of results per identity bin, ready for plotting.
    """
    bins = assign_identity_bins(max_identities)
    results = []

    for (lo, hi, label), mask in zip(IDENTITY_BINS, bins):
        print(f"\nEvaluating bin {label} (n={mask.sum()})...")

        probe_result = evaluate_probe_on_bin(probe, test_embeddings, test_labels, mask, device)
        blast_result = evaluate_blast_on_bin(blast_identities, test_labels, mask)

        # WAIC: run SGLD on training data, report per-bin test loss only
        # (WAIC is a property of the training fit, not the test bin)
        # We report test loss here; WAIC is computed once for the full model
        result = {
            "bin_label": label,
            "identity_lo": lo,
            "identity_hi": hi,
            "identity_midpoint": (lo + hi) / 2,
            "n_test": probe_result["n"],
            # Probe
            "probe_auroc": probe_result["auroc"],
            "probe_loss": probe_result["loss"],
            # BLAST
            "blast_sensitivity": blast_result["sensitivity"],
            "blast_specificity": blast_result.get("specificity", float("nan")),
            # LLC reference (same across all bins — it's at trained weights)
            "llc": llc_value,
            "llc_std": llc_std,
        }
        results.append(result)
        print(f"  Probe AUROC:      {probe_result['auroc']:.3f}")
        print(f"  BLAST sensitivity:{blast_result['sensitivity']:.3f}")

    return {"bins": results, "llc": llc_value, "llc_std": llc_std}


# ---------------------------------------------------------------------------
# SLT theoretical prediction (overlay on Figure 2)
# ---------------------------------------------------------------------------

def slt_generalization_bound(
    llc: float,
    n_train: int,
    identity_values: np.ndarray,
    scaling_constant: float = 1.0,
) -> np.ndarray:
    """
    SLT theoretical generalization error scales as lambda * log(n) / n.

    Here we use identity as a proxy for effective n (lower identity = fewer
    similar training examples = effectively smaller n).

    This is a qualitative overlay: we fit scaling_constant empirically from
    the high-identity bins, then project the SLT bound to low-identity bins.

    generalization_error(identity) ≈ C * lambda * log(n * identity) / (n * identity)

    Returns predicted AUROC degradation (1 - predicted error), for overlay.
    """
    effective_n = n_train * identity_values
    # Avoid log(0)
    effective_n = np.clip(effective_n, 1, None)
    predicted_error = scaling_constant * llc * np.log(effective_n) / effective_n
    return 1.0 - np.clip(predicted_error, 0, 1)
