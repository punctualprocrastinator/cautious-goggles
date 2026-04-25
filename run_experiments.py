"""
run_experiments.py
Full experimental pipeline for the hackathon project.

Usage:
  python run_experiments.py --layer 18 --device cpu
  python run_experiments.py --layer 18 --device cuda --skip-validation

Days 1-2: run with --validate-only to confirm LLC hyperparameters
Days 7-8: run full pipeline
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from compute_llc import (
    ToxinProbe,
    EmbeddingDataset,
    train_probe,
    estimate_llc,
    estimate_waic,
    validate_on_known_rlct,
)
from generalization_curve import run_generalization_curve, slt_generalization_bound


# ---------------------------------------------------------------------------
# LLC hyperparameter grid (run sensitivity analysis on Day 7)
# ---------------------------------------------------------------------------

LLC_HYPERPARAMS = [
    # (sgld_lr, beta, gamma, n_chains, n_draws, label)
    (1e-5, 0.1,  100.0, 5, 200, "default"),
    (1e-5, 0.01, 100.0, 5, 200, "low_beta"),
    (1e-6, 0.1,  100.0, 5, 200, "low_lr"),
    (1e-5, 0.1,  500.0, 5, 200, "high_gamma"),
    (1e-5, 0.1,  100.0, 5, 500, "more_draws"),
]


def run_llc_sensitivity(
    probe: ToxinProbe,
    embeddings: np.ndarray,
    labels: np.ndarray,
    loss_at_w_star: float,
    device: str = "cpu",
) -> dict:
    """
    Run LLC estimation across multiple hyperparameter settings.
    A stable LLC (low variance across settings) = trustworthy estimate.
    """
    results = {}
    for sgld_lr, beta, gamma, n_chains, n_draws, label in LLC_HYPERPARAMS:
        print(f"\n[LLC sensitivity] config={label}")
        r = estimate_llc(
            probe, embeddings, labels, loss_at_w_star,
            n_chains=n_chains, n_draws=n_draws,
            sgld_lr=sgld_lr, beta=beta, gamma=gamma,
            device=device,
        )
        results[label] = {
            "llc_mean": r["llc_mean"],
            "llc_std": r["llc_std"],
            "loss_mean": r["loss_mean"],
        }
        print(f"  LLC = {r['llc_mean']:.4f} ± {r['llc_std']:.4f}")

    # Check stability: all estimates within 20% of each other
    llc_values = [v["llc_mean"] for v in results.values()]
    spread = (max(llc_values) - min(llc_values)) / np.mean(llc_values)
    print(f"\nLLC spread across configs: {spread:.1%}")
    if spread < 0.20:
        print("✓ LLC is stable across hyperparameter settings.")
    else:
        print("✗ LLC is sensitive to hyperparameters. Use 'more_draws' config.")

    return results


# ---------------------------------------------------------------------------
# Layer sweep
# ---------------------------------------------------------------------------

def run_layer_sweep(
    embeddings_by_layer: dict[int, np.ndarray],  # layer_id -> (N, D) embeddings
    labels: np.ndarray,
    device: str = "cpu",
) -> dict:
    """
    Train a probe at each ESM-2 layer and report AUROC on natural toxin test set.
    Returns best layer for subsequent experiments.
    """
    results = {}
    best_auroc = 0
    best_layer = None

    for layer, embs in sorted(embeddings_by_layer.items()):
        # Stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            embs, labels, test_size=0.2, stratify=labels, random_state=42
        )
        probe, loss_star = train_probe(X_train, y_train, embed_dim=embs.shape[1], device=device)

        probe.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_test, dtype=torch.float32).to(device)
            logits = probe(X_t).cpu().numpy()
            probs = 1 / (1 + np.exp(-logits))   # sigmoid

        auroc = roc_auc_score(y_test, probs)
        results[layer] = {"auroc": auroc, "loss_at_w_star": loss_star}
        print(f"  Layer {layer:2d}: AUROC = {auroc:.4f}")

        if auroc > best_auroc:
            best_auroc = auroc
            best_layer = layer

    print(f"\nBest layer: {best_layer} (AUROC = {best_auroc:.4f})")
    return {"layer_results": results, "best_layer": best_layer}


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def main(args):
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    device = args.device

    # -----------------------------------------------------------------------
    # Step 0: Validate LLC hyperparameters (Days 1-2)
    # -----------------------------------------------------------------------
    if args.validate_only or not args.skip_validation:
        print("=" * 60)
        print("STEP 0: LLC validation on toy logistic regression")
        print("=" * 60)
        validate_on_known_rlct(device=device)
        if args.validate_only:
            return

    # -----------------------------------------------------------------------
    # Step 1: Load pre-computed embeddings
    # Assumes extract_esm2.py has been run and saved:
    #   embeddings/natural_toxins_layer{L}.npy
    #   embeddings/controls_layer{L}.npy
    #   embeddings/redesigns_layer{L}.npy
    #   embeddings/blast_identities.npy   (max BLAST identity of test seqs to training)
    #   embeddings/sequence_identities.npy (max seq identity of test seqs to training)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 1: Load embeddings")
    print("=" * 60)

    layer = args.layer
    emb_dir = Path("embeddings")

    toxin_embs = np.load(emb_dir / f"natural_toxins_layer{layer}.npy")
    control_embs = np.load(emb_dir / f"controls_layer{layer}.npy")
    redesign_embs = np.load(emb_dir / f"redesigns_layer{layer}.npy")

    # Labels: 1 = toxin, 0 = non-toxic
    toxin_labels = np.ones(len(toxin_embs))
    control_labels = np.zeros(len(control_embs))
    redesign_labels = np.ones(len(redesign_embs))   # redesigns ARE functional toxins

    # Training set: natural toxins + controls
    X_all = np.concatenate([toxin_embs, control_embs], axis=0)
    y_all = np.concatenate([toxin_labels, control_labels], axis=0)

    X_train, X_test_nat, y_train, y_test_nat = train_test_split(
        X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
    )

    # Full test set for generalization curve: nat + redesigns
    X_test_all = np.concatenate([X_test_nat, redesign_embs], axis=0)
    y_test_all = np.concatenate([y_test_nat, redesign_labels], axis=0)

    # Identity arrays (computed by data/compute_identities.py)
    # For nat test sequences: their identity to any training sequence
    # For redesigns: their identity to any training toxin (should be low!)
    seq_identities = np.load(emb_dir / "sequence_identities.npy")  # shape: (len(X_test_all),)
    blast_identities = np.load(emb_dir / "blast_identities.npy")   # shape: (len(X_test_all),)

    print(f"Training set: {len(X_train)} sequences")
    print(f"Natural test: {len(X_test_nat)} sequences")
    print(f"Redesigns:    {len(redesign_embs)} sequences")
    print(f"Redesign identity to training (mean): {blast_identities[-len(redesign_embs):].mean():.2f}")

    # -----------------------------------------------------------------------
    # Step 2: Train probe at best layer
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"STEP 2: Train probe at layer {layer}")
    print("=" * 60)

    probe, loss_at_w_star = train_probe(
        X_train, y_train,
        embed_dim=X_train.shape[1],
        device=device,
    )

    # Quick AUROC on natural test set
    probe.eval()
    with torch.no_grad():
        X_t = torch.tensor(X_test_nat, dtype=torch.float32).to(device)
        probs_nat = torch.sigmoid(probe(X_t)).cpu().numpy()
    auroc_natural = roc_auc_score(y_test_nat, probs_nat)
    print(f"AUROC on natural test set: {auroc_natural:.4f}")

    # -----------------------------------------------------------------------
    # Step 3: LLC estimation + sensitivity analysis
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 3: LLC estimation")
    print("=" * 60)

    llc_results = estimate_llc(
        probe, X_train, y_train, loss_at_w_star,
        n_chains=5, n_draws=200,
        sgld_lr=1e-5, beta=0.1, gamma=100.0,
        device=device,
    )
    print(f"LLC = {llc_results['llc_mean']:.4f} ± {llc_results['llc_std']:.4f}")
    print(f"Expected for regular model (d/2): {X_train.shape[1] / 2:.1f}")
    print(f"Ratio LLC / (d/2): {llc_results['llc_mean'] / (X_train.shape[1] / 2):.3f}")
    print(f"  (< 1 = degenerate/sparse solution; ~ 1 = full-rank = sequence memorisation)")

    # Sensitivity analysis
    if not args.skip_sensitivity:
        llc_sensitivity = run_llc_sensitivity(probe, X_train, y_train, loss_at_w_star, device)
        with open(results_dir / "llc_sensitivity.json", "w") as f:
            json.dump(llc_sensitivity, f, indent=2)

    # -----------------------------------------------------------------------
    # Step 4: WAIC
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 4: WAIC estimation")
    print("=" * 60)

    waic_results = estimate_waic(
        probe, X_train, y_train,
        n_chains=5, n_draws=200,
        sgld_lr=1e-5, gamma=100.0,
        device=device,
    )
    print(f"WAIC = {waic_results['wbic']:.4f}")
    print(f"(WAIC ≈ free energy; lower = better generalisation)")

    # -----------------------------------------------------------------------
    # Step 5: Generalization curve
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("STEP 5: Generalization curve")
    print("=" * 60)

    gen_curve = run_generalization_curve(
        probe=probe,
        test_embeddings=X_test_all,
        test_labels=y_test_all,
        max_identities=seq_identities,
        blast_identities=blast_identities,
        llc_value=llc_results["llc_mean"],
        llc_std=llc_results["llc_std"],
        train_embeddings=X_train,
        train_labels=y_train,
        device=device,
    )

    # SLT theoretical overlay
    identity_midpoints = np.array([b["identity_midpoint"] for b in gen_curve["bins"]])
    slt_predicted = slt_generalization_bound(
        llc=llc_results["llc_mean"],
        n_train=len(X_train),
        identity_values=identity_midpoints,
    )

    for b, slt_pred in zip(gen_curve["bins"], slt_predicted):
        b["slt_predicted_auroc"] = float(slt_pred)

    # -----------------------------------------------------------------------
    # Save all results
    # -----------------------------------------------------------------------
    all_results = {
        "layer": layer,
        "auroc_natural_test": float(auroc_natural),
        "llc": llc_results,
        "waic": waic_results,
        "generalization_curve": gen_curve,
        "n_train": len(X_train),
        "n_redesigns": len(redesign_embs),
        "embed_dim": X_train.shape[1],
    }

    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(results_dir / f"results_layer{layer}.json", "w") as f:
        json.dump(all_results, f, default=convert, indent=2)

    print(f"\nAll results saved to {results_dir}/results_layer{layer}.json")

    # -----------------------------------------------------------------------
    # Print summary table
    # -----------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"{'Bin':<12} {'n':>5} {'Probe AUROC':>12} {'BLAST Sens':>12} {'SLT pred':>10}")
    print("-" * 55)
    for b in gen_curve["bins"]:
        print(
            f"{b['bin_label']:<12} "
            f"{b['n_test']:>5} "
            f"{b['probe_auroc']:>12.3f} "
            f"{b['blast_sensitivity']:>12.3f} "
            f"{b.get('slt_predicted_auroc', float('nan')):>10.3f}"
        )
    print(f"\nLLC at w*: {llc_results['llc_mean']:.3f} ± {llc_results['llc_std']:.3f}")
    print(f"WAIC:      {waic_results['wbic']:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=18,
                        help="ESM-2 layer to use (1,9,18,24,30,33 for 650M)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="torch device (cpu or cuda)")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only run LLC validation on toy data then exit")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip toy LLC validation (if already confirmed)")
    parser.add_argument("--skip-sensitivity", action="store_true",
                        help="Skip LLC sensitivity analysis (saves ~30 min)")
    args = parser.parse_args()
    main(args)
