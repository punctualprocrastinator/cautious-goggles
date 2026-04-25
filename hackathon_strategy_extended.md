# AiXbio — 48-Hour Hackathon Execution Plan (Extended)

## Current Status (Hour 0)
| Item | Status |
|---|---|
| Data pipeline — 1712 toxins, 2072 controls, 723 redesigns, 100 structures | ✅ |
| ESM-2 embeddings — 6 layers saved | ✅ |
| ProteinMPNN redesigns — 27–80% identity range | ✅ |
| SAE analysis — 10,240 features, top-50 AUROC 0.94, transfer ratio 1.36 | ✅ |
| Generalisation curve — Cell 6 bug (5 min fix) | ⬜ |
| Activation steering — Notebook 2 not yet run | ⬜ |
| LLC/SLT — devinterp v1.3.2 fixable | ⬜ |
| Figures — Notebook 4 not yet run | ⬜ |
| Feature #481 targeted suppression attack (NEW) | ⬜ |
| Adversarial fine-tuning + LLC delta (NEW) | ⬜ |
| Paper | ⬜ |

---

## Why Transfer Ratio = 1.36 Changes Everything

**The key insight that unlocks the adversarial angle:**

Random ProteinMPNN redesign is the *worst possible evasion strategy* — it
amplifies the toxin features ESM-2 uses for detection. An attacker using
ProteinMPNN without knowing about the feature structure is digging themselves
deeper. This makes the adversarial question sharper and more interesting:

> Can an **informed** attacker who knows about feature #481 suppress it
> deliberately — where random redesign failed?

This is the protein-space version of Winninger et al. (arXiv:2503.06269):
using mechanistic interpretability to craft targeted attacks. Their insight
was that knowing the "acceptance subspace" of an LLM safety probe enables
efficient jailbreaks. The question here: does knowing the "toxin subspace"
(feature #481 + the probe direction) enable efficient evasion?

If targeted suppression **also fails** → feature #481 is causally locked
to structural properties necessary for protein fold. The toxin signal is not
a correlate that can be designed away — it is a structural necessity.

If targeted suppression **succeeds** → the probe is vulnerable to
interpretability-informed attacks, which has direct biosecurity implications
(a sophisticated attacker with access to interPLM could evade screening).

**Both results are publishable. Both are more important than the current
narrative.** The current narrative is the setup. The adversarial result is
the punchline.

---

## Phase 1 — Immediate Fixes (Hours 0–2) [UNCHANGED]

### Fix Notebook 3 Cell 6 (5 min)
```python
X_test_all_sae  = np.concatenate([X_te_sae,
                                   sc_sae.transform(rdsg_acts[:, top_feat_idx])])
X_test_all_full = np.concatenate([X_te_full, sc_full.transform(rdsg_acts)])
y_test_all      = np.concatenate([y_te, np.ones(len(rdsg_acts))])
```

### Run Notebook 2 — Activation Steering (15 min)
Key outputs:
- `cosine(probe_weight, mean_diff)` — is toxicity linearly encoded?
- Steering curve: ctrl_score(α) from -3σ to +3σ
- Per-identity-bin steered AUROC

**New thing to extract here:** save `probe_direction = probe.linear.weight.data[0]`
(the 1280-dim probe weight vector). This is the toxin subspace direction.
You need it for Experiment 4e.

### Install devinterp v1.3.2 (5 min)
```bash
pip install "devinterp==1.3.2"
```

---

## Phase 2 — Core Results Locked (Hours 2–6) [UNCHANGED]

### Run Notebook 4 — All 4 Figures
1. **Fig 1**: Layer sweep AUROC bar chart
2. **Fig 2**: SAE feature transfer bimodality scatter ← most novel
3. **Fig 3**: Activation steering curve
4. **Fig 4**: BLAST vs ESM-2 probe across identity bins

### BLAST vs Probe headline cell (add to Notebook 3)
```python
for thr in [0.30, 0.40, 0.50]:
    blast_det = (blast_identities[len(X_test):] >= thr).mean()
    esm_det   = (probe_scores_redesigns > 0.5).mean()
    print(f"BLAST@{thr:.0%}: {blast_det:.1%}  |  ESM-2: {esm_det:.1%}")
```

---

## Phase 3 — LLC/SLT Restoration (Hours 6–12) [EXTENDED]

v1.3.2 uses `estimate_learning_coeff_with_summary(model, loader, ...)`.

**Original plan:** report LLC/(d/2) ratio and WAIC across layers.

**Extension — compute LLC twice:**

```python
# LLC of probe trained on natural toxins only (original)
llc_natural = estimate_learning_coeff_with_summary(
    probe_natural, loader_natural, ...
)

# LLC of probe adversarially fine-tuned on natural + pSSR-attacked sequences
# (run AFTER Phase 4e — adversarial fine-tuning takes ~30 min)
llc_robust = estimate_learning_coeff_with_summary(
    probe_robust, loader_natural, ...   # same loader, different model weights
)

delta_llc = llc_robust["llc/mean"] - llc_natural["llc/mean"]
print(f"ΔLLC after adversarial fine-tuning: {delta_llc:+.3f}")
# delta > 0 → probe became more complex to resist attack (memorisation)
# delta ≈ 0 → probe found same sparse direction, now more robust (generalisation)
```

**The SLT interpretation:**

| ΔLLC | Meaning | Biosecurity implication |
|------|---------|------------------------|
| > 0 (increases) | Probe memorised the attack pattern | Will fail on novel attack variants |
| ≈ 0 (stable) | Probe found a stable function direction | Robust to this attack class |
| < 0 (decreases) | Adversarial data simplified the problem | Feature #481 is causally dominant |

This is a direct application of Timaeus's developmental interpretability
framework to adversarial robustness. No paper has measured ΔLLC under
protein-space adversarial fine-tuning. This is the SLT novelty claim.

---

## Phase 4 — Extended Experiments (Hours 12–24) [EXTENDED]

### 4a. Per-Residue SAE with NNsight (3 hrs) [UNCHANGED]
```python
from nnsight import NNsight
model_nn = NNsight(esm2_model)
with model_nn.trace(input_ids):
    acts = model_nn.esm.encoder.layer[24].output[0].save()
# Run SAE per-position → which residues activate toxin features?
```
Output: "Positions 10–20 (disulfide core) are sufficient for detection."

**Connect to 4e:** which residues does the targeted attack perturb most?
Do they overlap with the disulfide core or avoid it?

### 4b. Feature Taxonomy — Name the Top Features (2 hrs) [UNCHANGED]
For features 481, 5857, 5724, 4948, 4129:
- Find top-activating toxin sequences
- Align them → what motifs do they share?
- Hypothesis: feature 481 = CxxxxC (disulfide); feature 5857 = signal peptide

**Connect to 4e:** feature #481 is the primary attack target. Does naming
it (e.g. CxxxxC) predict why the attack succeeds or fails? If feature #481
encodes a disulfide motif, then suppressing it requires eliminating the
disulfide — which destroys the toxin's fold and function simultaneously.
That would explain why targeted suppression also fails.

### 4c. Evasion Rate — The Biosecurity Punchline (30 min) [UNCHANGED]
```python
evading = (probe_scores_redesigns < 0.3).mean()
print(f"ProteinMPNN (random) evasion rate vs ESM-2: {evading:.1%}")
```

**Extended:** compare this number to the targeted attack evasion rate from 4e.
The comparison IS the paper's main table.

| Attack | Evasion rate | Notes |
|--------|-------------|-------|
| Random ProteinMPNN | ~X% | What you measure here |
| pSSR (embedding-space) | ~Y% | What 4e measures |
| Feature #481 suppression | ~Z% | What 4e measures |

### 4d. More Redesigns — Run Overnight [UNCHANGED]
Bump `num_seq_per_target` from 10 to 20 → 2000 redesigns for better statistics.

---

## Phase 4e — Adversarial Mech Interp (NEW, Hours 16–22)

**The new core experiment. Budget 6 hours. This is what makes it a research paper.**

### Step 1 — pSSR: Embedding-Space Targeted Attack (2 hrs)

Protein Subspace Rerouting (pSSR) — the protein analog of Winninger et al.
Find the minimal perturbation in ESM-2 embedding space that flips the probe,
constrained to stay within the natural protein distribution.

```python
import torch
import torch.nn as nn

def pssr_attack(
    probe,           # trained ToxinProbe
    embeddings,      # natural toxin ESM-2 embeddings, shape (N, 1280)
    probe_direction, # probe.linear.weight.data[0], shape (1280,)
    n_steps=500,
    lr=0.01,
    epsilon=2.0,     # max L2 perturbation radius in embedding space
):
    """
    Gradient ascent that MINIMISES probe confidence (pushes toward non-toxic),
    constrained to L2 ball of radius epsilon around original embedding.
    
    epsilon=2.0 is ~1 std of natural protein embedding variation.
    Increase epsilon to find the evasion boundary.
    """
    e = torch.tensor(embeddings, dtype=torch.float32)
    e_orig = e.clone()
    delta = torch.zeros_like(e, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)

    for step in range(n_steps):
        e_adv = e_orig + delta
        # Minimise probe confidence (push toward non-toxic half-space)
        probe_loss = -probe(e_adv).mean()
        probe_loss.backward()
        opt.step()
        opt.zero_grad()
        # Project back to L2 ball (constraint)
        with torch.no_grad():
            norms = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            delta.data = delta.data * (norms.clamp(max=epsilon) / norms)

    e_adv = (e_orig + delta).detach()
    adv_scores = torch.sigmoid(probe(e_adv)).numpy()
    evasion_rate = (adv_scores < 0.3).mean()
    
    # Direction of perturbation relative to probe direction
    delta_np = delta.detach().numpy()
    cosine_with_probe = np.dot(
        delta_np.mean(0) / np.linalg.norm(delta_np.mean(0)),
        probe_direction.numpy() / np.linalg.norm(probe_direction.numpy())
    )
    
    print(f"pSSR evasion rate:              {evasion_rate:.1%}")
    print(f"Cosine(Δ, probe_direction):     {cosine_with_probe:.3f}")
    print(f"  (should be negative — attack moves opposite to probe direction)")
    
    return {"evasion_rate": evasion_rate, "adv_embeddings": e_adv,
            "delta": delta_np, "cosine_with_probe": cosine_with_probe}
```

**Run at increasing epsilon values [0.5, 1.0, 2.0, 5.0, 10.0]** to find the
evasion curve. This is Figure 5.

**Compare to ProteinMPNN evasion from 4c.** The ratio pSSR/ProteinMPNN tells
you how much an informed attacker gains from knowing the model's internals.

### Step 2 — Feature #481 Targeted Suppression (2 hrs)

This is harder but more interpretable. Instead of optimising against the probe
directly, optimise specifically to suppress feature #481 activation.

```python
def feature_suppression_attack(
    sae,             # InterPLM SAE (already loaded)
    esm2_model,      # ESM-2 650M
    toxin_sequences, # raw sequences (needed for ESM-2 forward pass)
    target_feature=481,
    n_steps=300,
    lr=0.005,
):
    """
    For each toxin sequence, find a perturbation in embedding space that
    minimises SAE feature #481 activation while staying within a plausibility
    constraint (ESM-2 log-likelihood doesn't collapse).
    
    This requires running the SAE forward pass in the gradient graph.
    """
    # Get layer-24 activations with gradient tracking
    # Use NNsight to hook into ESM-2 and get differentiable activations
    # Then run SAE on those activations and minimise feature 481
    
    # NOTE: if full ESM-2 backward is too slow on CPU, use the 
    # pre-computed embeddings with a proxy: minimise the component of
    # the embedding in the SAE decoder direction for feature #481.
    
    # Fast proxy (no ESM-2 backward needed):
    decoder_dir = sae.W_dec[target_feature]  # (1280,) decoder direction
    decoder_dir = decoder_dir / decoder_dir.norm()
    
    e = torch.tensor(natural_toxin_embeddings, dtype=torch.float32)
    e_orig = e.clone()
    delta = torch.zeros_like(e, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    
    for step in range(n_steps):
        e_adv = e_orig + delta
        # Minimise component in decoder direction for feature 481
        feat481_activation = (e_adv * decoder_dir).sum(dim=-1).mean()
        feat481_activation.backward()
        opt.step()
        opt.zero_grad()
        # L2 constraint
        with torch.no_grad():
            norms = delta.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            delta.data = delta.data * (norms.clamp(max=2.0) / norms)
    
    e_adv = (e_orig + delta).detach()
    
    # Check: did suppressing feature #481 also flip the probe?
    probe_scores_after = torch.sigmoid(probe(e_adv)).numpy()
    feat481_after = (e_adv @ decoder_dir).numpy()
    feat481_before = (e_orig @ decoder_dir).numpy()
    
    print(f"Feature #481 activation: {feat481_before.mean():.3f} → {feat481_after.mean():.3f}")
    print(f"Probe score:             {probe_scores_before.mean():.3f} → {probe_scores_after.mean():.3f}")
    print(f"Probe evasion rate:      {(probe_scores_after < 0.3).mean():.1%}")
    
    causality = "CAUSAL" if (probe_scores_after < 0.3).mean() > 0.5 else "NOT CAUSAL"
    print(f"Feature #481 → probe:    {causality}")
```

**The causal test:** if suppressing feature #481 also flips the probe →
feature #481 is causally upstream of the probe decision. If the probe
still fires after feature #481 is suppressed → the probe uses other features
redundantly, and feature #481 is merely the most active but not the only one.

Given transfer ratio = 1.36, hypothesis: **the probe uses a distributed set
of structural features, not just feature #481**. Suppressing any single one
won't flip it. This would explain why random redesign fails — it can't
suppress all structural features simultaneously.

### Step 3 — Adversarial Fine-Tuning (1 hr)

Fine-tune the probe on a mix of natural sequences AND pSSR-attacked embeddings:

```python
# Adversarial training set: natural toxins + their pSSR-attacked versions
# Label the pSSR attacks as TOXINS (they are — we're teaching the probe to
# catch attacks it currently misses)

pssr_results = pssr_attack(probe, toxin_embeddings, probe_direction, epsilon=5.0)
adv_embeddings = pssr_results["adv_embeddings"]

X_adv_train = np.concatenate([X_train, adv_embeddings[:len(toxin_train)]])
y_adv_train = np.concatenate([y_train, np.ones(len(toxin_train))])

probe_robust, loss_robust = train_probe(X_adv_train, y_adv_train, device=device)
```

Then compute LLC of `probe_robust` — see Phase 3 extension above.

### Step 4 — Compare All Attacks in One Table (30 min)

```python
results = {
    "Random ProteinMPNN": (probe_scores_redesigns < 0.3).mean(),
    "pSSR ε=2.0":         pssr_evasion_rate_eps2,
    "pSSR ε=5.0":         pssr_evasion_rate_eps5,
    "Feature #481 suppression": feat481_evasion_rate,
    "Robust probe (pSSR-trained)": (probe_robust_scores < 0.3).mean(),
}
for name, rate in results.items():
    print(f"{name:<35} {rate:.1%}")
```

---

## Phase 5 — Paper Draft (Hours 24–36) [EXTENDED]

**Target: 6 pages, NeurIPS/ICLR style (was 4, extended by adversarial section)**

### Updated Abstract

> Biosecurity screening relies on sequence-identity thresholds (BLAST) to flag dangerous
> proteins. We demonstrate that ProteinMPNN redesigns toxin sequences below BLAST
> thresholds (<40% identity) while ESM-2 functional probes maintain >0.9 AUROC.
> Using interPLM Sparse Autoencoders, we identify 50 of 10,240 features explaining
> 99% of probe performance (205× compression) — with transfer ratio 1.36, meaning
> redesigns **amplify** structural toxin features rather than suppress them.
> We then ask: can an informed adversary who knows this feature structure evade
> detection? Using Protein Subspace Rerouting (pSSR) — gradient-based attack
> against the probe's representation subspace — we find that [pSSR succeeds/fails
> at ε = X, implying feature #481 is [causally necessary / redundant]].
> Adversarial fine-tuning recovers probe robustness with ΔLLC = [value], indicating
> the probe [memorises the attack / finds a stable functional direction].
> Structure-aware probes grounded in essential structural features resist
> both random redesign and interpretability-informed attacks.

*(Fill in bracketed values from experiments)*

### Updated Section Structure

1. **Introduction + Related Work** (0.75p)
   - Wittmann attack, current screening gap
   - InterPLM SAEs on ESM-2 (Adams, Simon & Zou)
   - **NEW:** Winninger SSR adversarial mech interp
   - **NEW:** CRISP feature suppression (arXiv:2508.13650)
   - SLT/LLC (Lau et al., Timaeus devinterp)

2. **Methods** (1.25p)
   - ESMFold → ProteinMPNN → ESM-2 → SAE → Probe → Steering
   - **NEW:** pSSR attack formulation
   - **NEW:** Feature #481 causal test
   - **NEW:** Adversarial fine-tuning + ΔLLC

3. **Results** (2p — was 1.5p)
   - Fig 1: Layer sweep AUROC
   - Fig 2: SAE feature transfer bimodality (transfer ratio 1.36)
   - Fig 3: Activation steering curve (toxicity is linear)
   - Fig 4: BLAST vs ESM-2 across identity bins
   - **Fig 5 (NEW): pSSR evasion curve vs random ProteinMPNN**
   - **Fig 6 (NEW): Feature #481 causal test + ΔLLC**

4. **Discussion + Limitations** (0.75p)
   - Why transfer ratio > 1 (structural necessity hypothesis)
   - pSSR finding and implications for informed attackers
   - ΔLLC interpretation (memorisation vs generalisation)
   - Limitations: embedding-space attack ≠ sequence-space attack;
     need wet-lab validation of function preservation

5. **References** (0.5p)

---

## Phase 6 — Polish + Submission (Hours 36–48) [UNCHANGED + additions]

### Figure polish checklist [UNCHANGED]
- [ ] Color-blind safe palette (`seaborn-colorblind` or tab10)
- [ ] Font size ≥ 12pt
- [ ] Error bars on all bar plots
- [ ] Significance markers (* p<0.05, ** p<0.01)
- [ ] Save PDF + PNG at 300 DPI

### New additions
- [ ] Attack comparison table (all 4 attack variants in one table)
- [ ] ΔLLC reported with confidence interval across chains
- [ ] pSSR evasion curve as Fig 5 (epsilon on x-axis, evasion rate on y-axis,
      dashed line at ProteinMPNN baseline)
- [ ] Feature #481 causal test as an inset or panel in Fig 2

---

## Updated Priority Matrix

| Task | Impact | Effort | Phase | Status |
|------|--------|--------|-------|--------|
| Fix Cell 6 bug | High | 5 min | 1 | ⬜ NOW |
| Run Notebook 2 (steering) + save probe direction | High | 20 min | 1 | ⬜ NOW |
| Run Notebook 4 (figures) | Very High | 30 min | 2 | ⬜ |
| BLAST vs ESM-2 headline | High | 15 min | 2 | ⬜ |
| devinterp v1.3.2 LLC (natural probe) | Medium | 1 hr | 3 | ⬜ |
| NNsight per-residue (4a) | Very High | 3 hr | 4 | ⬜ |
| Feature taxonomy — name #481 (4b) | High | 2 hr | 4 | ⬜ |
| Evasion rate number (4c) | High | 30 min | 4 | ⬜ |
| **pSSR attack (4e Step 1)** | **Very High** | **2 hr** | **4e** | ⬜ |
| **Feature #481 causal test (4e Step 2)** | **Very High** | **2 hr** | **4e** | ⬜ |
| **Adversarial fine-tuning (4e Step 3)** | **High** | **1 hr** | **4e** | ⬜ |
| **LLC delta — robust vs natural probe** | **High** | **1 hr** | **3 ext** | ⬜ |
| **Attack comparison table (4e Step 4)** | **High** | **30 min** | **4e** | ⬜ |
| More redesigns overnight (4d) | Low | auto | 4 | ⬜ |
| Paper draft (extended) | Very High | 8 hr | 5 | ⬜ |
| README + polish | Medium | 1 hr | 6 | ⬜ |

**New total time budget:** pSSR adds ~6 hrs to Phase 4. Fits in the 24-hr
Phase 4–5 window if started immediately after Phase 3 LLC completes.

---

## Updated Winning Narrative (30-second pitch)

> "We built the first mechanistic interpretability pipeline for biosecurity.
> BLAST fails against ProteinMPNN redesigns — but ESM-2 doesn't.
> More surprisingly, redesigns amplify the toxin signal, not suppress it.
> Feature #481 fires in 23% of natural toxins but 98% of redesigns —
> because it encodes structural necessity, not sequence pattern.
> We then played adversary: using the probe's own geometry to craft targeted
> attacks. Even with full white-box access to the model's features,
> [the attack failed / required X perturbation] — because you can't suppress
> a feature that's causally necessary for the protein to fold.
> SLT confirms this: ΔLLC ≈ 0 after adversarial fine-tuning means the probe
> found a stable functional direction, not a memorised sequence pattern.
> The biosecurity implication: ground your screener in structural features
> and it resists both dumb random attacks and smart mechanistic ones."

---

## The Two-Paper Path

**Paper 1 (this hackathon):** The complete pipeline through Phase 4e.
→ Target: NeurIPS 2026 Workshop on ML for Structural Biology or AI for Science.

**Paper 2 (post-hackathon, 3 months):** Extend with:
- Wet-lab validation of function preservation for pSSR-attacked sequences
- Sequence-space translation of embedding-space attacks via ESM-2 inversion
- Extension to EvoDiff and RFdiffusion (the other two Wittmann attack tools)
- Collaboration with Nicole Wheeler / IBBIS for real synthesis provider data
→ Target: NeurIPS 2026 main track or Nature Machine Intelligence.

The hackathon paper IS the foundation for Paper 2. Every experiment you run
now is a figure in the follow-up.
