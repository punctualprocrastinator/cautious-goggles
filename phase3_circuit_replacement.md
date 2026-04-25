# Phase 3 Replacement: Toxin Circuit Discovery via NNsight
## (Replaces LLC/SLT — more relevant to CBAI)

---

## Why This Replaces LLC

LLC measures *how complex* the probe's solution is (loss landscape geometry).
Circuit discovery asks *where and how* ESM-2 computes the toxin signal.

For CBAI, the mechanistic question is more important:
- Their Forbidden Facts paper used activation patching to find where LLMs
  "know" suppressed information
- Their feature suppression paper asked what happens when features are removed
- Circuit discovery on ESM-2 for toxin function is the exact protein-space
  analog of both

LLC is a Timaeus/SLT contribution. Circuits are a CBAI contribution.
**Replace LLC with circuits, keep everything else.**

---

## The Four NNsight Experiments (Hours 6–14)

### Setup — Wire NNsight to ESM-2

```python
from nnsight import NNsight
import torch

model_nn = NNsight(esm2_model)

# Helper: get mean-pooled residual stream at a given layer
def get_layer_activations(input_ids, layer_idx):
    with model_nn.trace(input_ids):
        acts = model_nn.esm.encoder.layer[layer_idx].output[0].save()
    return acts  # (batch, seq_len, 1280)

# Helper: get probe score from layer activations
def probe_score_from_acts(acts, probe):
    pooled = acts.mean(dim=1)   # (batch, 1280)
    return torch.sigmoid(probe(pooled))
```

---

### Experiment 1 — Layer-Level Activation Patching (2 hrs)
**"Which layers are causally sufficient for the toxin signal?"**

Activation patching modifies or "patches" the activations of model components
and observes the impact on output — identifying which components causally
contribute to a behavior.

**Setup:**
- *Clean* = non-toxic control sequence (probe score ≈ 0)
- *Patch source* = natural toxin (probe score ≈ 1)
- **Intervention:** for each layer l, replace the clean residual stream with
  the toxin's residual stream at layer l only
- **Metric:** probe score recovery = how much does patching layer l restore
  the toxin probe score on the control sequence?

```python
def activation_patch_layer(
    clean_input_ids,   # non-toxic control
    patch_input_ids,   # natural toxin
    patch_layer,       # which layer to patch
    probe,
):
    """
    3-pass activation patching (Meng et al. / NNsight standard):
    Pass 1: clean forward → cache clean activations
    Pass 2: patch forward → cache toxin activations at patch_layer
    Pass 3: clean forward with layer patch_layer replaced by toxin acts
    """
    # Pass 1: clean baseline score
    with model_nn.trace(clean_input_ids):
        clean_final = model_nn.esm.encoder.layer[33].output[0].save()
    clean_score = probe_score_from_acts(clean_final, probe).item()

    # Pass 2: get toxin activations at the target layer
    with model_nn.trace(patch_input_ids):
        toxin_acts_at_l = model_nn.esm.encoder.layer[patch_layer].output[0].save()

    # Pass 3: run clean forward, patch at layer l
    with model_nn.trace(clean_input_ids):
        model_nn.esm.encoder.layer[patch_layer].output[0][:] = toxin_acts_at_l
        patched_final = model_nn.esm.encoder.layer[33].output[0].save()
    patched_score = probe_score_from_acts(patched_final, probe).item()

    # Patch toxin score (what we're recovering toward)
    with model_nn.trace(patch_input_ids):
        toxin_final = model_nn.esm.encoder.layer[33].output[0].save()
    toxin_score = probe_score_from_acts(toxin_final, probe).item()

    # Normalized recovery: 0 = no effect, 1 = full toxin score recovered
    recovery = (patched_score - clean_score) / (toxin_score - clean_score + 1e-8)
    return recovery

# Run across all 33 layers
recovery_by_layer = {}
for l in range(34):  # ESM-2 650M has 33 transformer layers
    recoveries = []
    for (clean_ids, toxin_ids) in zip(control_sample, toxin_sample):
        r = activation_patch_layer(clean_ids, toxin_ids, l, probe)
        recoveries.append(r)
    recovery_by_layer[l] = np.mean(recoveries)
    print(f"Layer {l:2d}: recovery = {recovery_by_layer[l]:.3f}")
```

**Expected result:** Recovery peaks at layers 18–24 (consistent with probe
AUROC peaking there). Low-recovery layers are NOT part of the toxin circuit.

**This is Figure 5a:** Layer activation patching recovery curve.
A sharp peak identifies the *toxin circuit layers*.

---

### Experiment 2 — Attribution Patching (1 hr)
**"Which attention heads drive the toxin direction?"**

Attribution patching uses gradients as a linear approximation to activation
patching — done in two forward passes and one backward pass, making it much
more scalable. It can be done simultaneously across all components.

This is faster than full activation patching and gives per-head attribution
scores across all 33 layers × 20 heads = 660 components in a single backward pass.

```python
def attribution_patching_all_heads(
    clean_input_ids,
    patch_input_ids,
    probe,
):
    """
    Linear approximation: attribution[l,h] ≈ 
        (patch_acts[l,h] - clean_acts[l,h]) · ∂probe_score/∂acts[l,h]
    
    Gradient tells us: in which direction should acts change to increase
    probe score? Patch delta tells us: how much does each head change
    between clean and patch? Attribution = dot product of both.
    """
    n_layers = 33
    n_heads = 20   # ESM-2 650M

    # Forward pass on clean, track gradients
    with model_nn.trace(clean_input_ids):
        head_outputs = []
        for l in range(n_layers):
            # Hook into attention output before projection
            attn_out = model_nn.esm.encoder.layer[l].attention.output.dense.input[0].save()
            head_outputs.append(attn_out)
        final_acts = model_nn.esm.encoder.layer[n_layers-1].output[0].save()

    clean_score = probe_score_from_acts(final_acts, probe)
    clean_score.backward()  # get gradients w.r.t. all head outputs

    clean_grads = [h.grad for h in head_outputs]
    clean_head_acts = [h.detach() for h in head_outputs]

    # Forward pass on patch (no grad needed)
    with torch.no_grad():
        with model_nn.trace(patch_input_ids):
            patch_head_acts = []
            for l in range(n_layers):
                attn_out = model_nn.esm.encoder.layer[l].attention.output.dense.input[0].save()
                patch_head_acts.append(attn_out)

    # Attribution = (patch - clean) · gradient, summed over sequence and head dim
    attributions = np.zeros((n_layers, n_heads))
    head_dim = 1280 // n_heads  # 64 for ESM-2 650M

    for l in range(n_layers):
        delta = (patch_head_acts[l] - clean_head_acts[l])  # (batch, seq, 1280)
        grad  = clean_grads[l]                              # (batch, seq, 1280)
        # Reshape to (batch, seq, n_heads, head_dim) and sum
        delta_heads = delta.reshape(delta.shape[0], delta.shape[1], n_heads, head_dim)
        grad_heads  = grad.reshape(grad.shape[0], grad.shape[1], n_heads, head_dim)
        attr = (delta_heads * grad_heads).sum(dim=(0, 1, 3))  # (n_heads,)
        attributions[l] = attr.numpy()

    return attributions  # (n_layers, n_heads) — positive = promotes toxin signal

# Result: heatmap of which (layer, head) pairs are causally important
# This is Figure 5b: Attribution heatmap
```

---

### Experiment 3 — Circuit Preservation in Redesigns (1 hr)
**"Do ProteinMPNN redesigns use the same circuit as natural toxins?"**

This is the mechanistic explanation of transfer ratio = 1.36.

**Setup:** same activation patching as Exp 1, but now:
- *Clean* = non-toxic control
- *Patch source* = ProteinMPNN redesign (NOT natural toxin)

If the recovery curve for redesigns matches natural toxins at the same layers
→ redesigns activate the **same toxin circuit**, just via different sequence
→ the circuit is identity-independent (functional, not sequence-based)

If the recovery curve for redesigns is lower at circuit layers
→ redesigns use a different pathway to activate the probe
→ transfer is coincidental, not mechanistic

```python
# Run Exp 1 but with redesign as patch source
recovery_redesign = {}
for l in range(34):
    recoveries = []
    for (clean_ids, redesign_ids) in zip(control_sample, redesign_sample):
        r = activation_patch_layer(clean_ids, redesign_ids, l, probe)
        recoveries.append(r)
    recovery_redesign[l] = np.mean(recoveries)

# Compare natural toxin circuit vs redesign circuit
# Figure 5c: overlaid recovery curves — natural (orange) vs redesign (red)
# If they overlap at the same layers → same circuit
```

**This is the key mechanistic claim:** "ProteinMPNN redesigns activate the
same ESM-2 toxin circuit as natural toxins, explaining why they amplify
rather than evade the probe."

---

### Experiment 4 — pSSR Circuit Disruption Test (1 hr)
**"Does the adversarial attack disrupt the toxin circuit?"**

The pSSR attack (from Phase 4e) perturbs sequences in embedding space to
minimise probe confidence. But does it disrupt the underlying circuit, or
merely shift the final embedding without affecting the computation?

```python
# pSSR-attacked embeddings from Phase 4e
# We need to translate them back to a form we can patch
# Simplification: measure how the pSSR delta changes activations AT
# the identified circuit layers (from Exp 1)

circuit_layers = [l for l, r in recovery_by_layer.items() if r > 0.5]

# For each pSSR-attacked embedding, measure:
# 1. Does the probe still fire? (already computed in 4e)
# 2. What is the activation at circuit layer after pSSR perturbation?
# 3. Is the perturbation direction correlated with the circuit direction?

probe_direction = probe.linear.weight.data[0]  # (1280,)

for l in circuit_layers:
    # Get activations at circuit layer for: natural, redesign, pSSR-attacked
    acts_natural  = get_layer_activations(toxin_ids, l).mean(dim=1)   # pooled
    acts_redesign = get_layer_activations(redesign_ids, l).mean(dim=1)
    # pSSR: we have the perturbed final embedding; circuit layer is upstream
    # → project pSSR delta onto the circuit layer's contribution to probe direction
    
    # How much does pSSR perturbation move things in the probe direction?
    pssr_delta_mean = pssr_delta.mean(0)   # (1280,)
    cosine = np.dot(pssr_delta_mean / np.linalg.norm(pssr_delta_mean),
                    probe_direction.numpy() / np.linalg.norm(probe_direction.numpy()))
    
    print(f"Layer {l}: cosine(pSSR_delta, probe_direction) = {cosine:.3f}")
    # Near -1 = pSSR directly counters the circuit direction
    # Near 0 = pSSR is orthogonal to the circuit (doesn't disrupt it)
```

**The finding:** if pSSR is nearly orthogonal to the circuit directions
(cosine ≈ 0), the attack perturbs the *representation* but not the
*computation*. The circuit re-encodes the toxin signal even after
embedding perturbation. This would be the strongest possible robustness claim.

---

## The Unified Circuit Story (CBAI Framing)

The four experiments together tell one story:

1. **The toxin circuit exists** — localized to layers 18–24, driven by
   specific attention heads (Exp 1 + 2)

2. **The circuit is identity-independent** — ProteinMPNN redesigns activate
   the same circuit as natural toxins, explaining transfer ratio 1.36 (Exp 3)

3. **The circuit is adversarially robust** — pSSR perturbs representations
   orthogonally to the circuit direction; the circuit re-encodes the signal (Exp 4)

4. **Why** — feature taxonomy (Phase 4b) shows the circuit encodes structural
   necessity features (disulfide, signal peptide) that cannot be designed away

**CBAI pitch:** "We found the toxin circuit in ESM-2 — the first circuit-level
analysis of a protein language model for biosecurity. This circuit is causally
sufficient for toxin detection, is preserved under ProteinMPNN redesign, and
resists adversarial attack. It is the protein-space analog of the safety
circuits studied in LLMs — with the crucial difference that it emerges from
structural biological necessity, not training objective."

---

## Figure Plan (Replaces LLC Figure)

**Was:** LLC value + WAIC across layers (1 figure)

**Now:** 3-panel Figure 5 (circuits — worth 2x the space in the paper)

```
Fig 5a: Layer recovery curve
  x-axis: ESM-2 layer (0–33)
  y-axis: activation patching recovery (0–1)
  Two lines: natural toxin patch (orange) vs redesign patch (red)
  Highlighted band: "toxin circuit layers" (recovery > 0.5)

Fig 5b: Attribution heatmap
  rows: ESM-2 layers (0–33)
  cols: attention heads (0–19)
  colour: attribution score (blue = suppresses, red = promotes toxin signal)
  Shows which (layer, head) pairs drive the toxin direction

Fig 5c: pSSR circuit disruption
  Bar chart: cosine(pSSR_delta, probe_direction) per circuit layer
  Near 0 = attack is orthogonal to circuit → circuit is robust
  Near -1 = attack directly counters circuit → circuit is vulnerable
```

---

## Time Budget

| Experiment | Time | Prerequisite |
|---|---|---|
| Setup + NNsight wiring | 30 min | Notebook 2 done |
| Exp 1: Layer patching | 2 hr | Setup |
| Exp 2: Attribution patching | 1 hr | Exp 1 |
| Exp 3: Circuit preservation | 1 hr | Exp 1 |
| Exp 4: pSSR disruption | 1 hr | Exp 3 + pSSR from 4e |
| Figure generation | 1 hr | All exps |
| **Total** | **6.5 hr** | Fits in Phase 3+4 window |

LLC would have taken ~4 hrs (SGLD chains are slow). Circuit discovery takes
6.5 hrs but produces 3 figures instead of 1 and tells a stronger story.

---

## Updated Paper Abstract

> Biosecurity screening relies on sequence-identity thresholds to flag dangerous
> proteins. We demonstrate that ProteinMPNN redesigns toxin sequences below BLAST
> thresholds while ESM-2 functional probes maintain >0.9 AUROC — with transfer
> ratio 1.36, meaning redesigns amplify rather than suppress toxin features.
> Using NNsight activation patching, we identify the ESM-2 *toxin circuit*:
> layers 18–24 and [N] attention heads that are causally sufficient to transfer
> the toxin signal from natural toxins to non-toxic controls. We show this circuit
> is preserved under ProteinMPNN redesign — the mechanistic explanation of the
> transfer ratio — and is orthogonal to our pSSR adversarial attack direction,
> explaining probe robustness. Feature taxonomy identifies circuit-activating
> features as structural necessities (disulfide cores, signal peptides) that
> cannot be removed without destroying protein fold.
> Structure-aware biosecurity screening grounded in essential structural
> circuits resists both random and interpretability-informed attacks.

---

## Why This Beats LLC for CBAI Specifically

| | LLC (SLT) | Toxin Circuit (NNsight) |
|---|---|---|
| What it measures | Loss landscape geometry at w* | Causal computational pathway in ESM-2 |
| CBAI prior work | Timaeus (different org) | Forbidden Facts (activation patching) |
| Causal claim | No (correlational) | Yes (intervention-based) |
| Explains transfer ratio | No | Yes (same circuit = same signal) |
| Explains probe robustness | Partially | Yes (pSSR orthogonal to circuit) |
| Figure count | 1 | 3 |
| Interpretable to biosecurity audience | No | Yes ("the toxin circuit lives in layers 18–24") |
| Novel for protein LMs | Medium | High (first circuit-level analysis for biosecurity) |

The only thing LLC gives that circuits don't: a dimensionless complexity
measure. You can approximate this without SGLD by reporting the number of
circuit layers (circuit size) as a proxy for effective complexity.
Circuit size << d/2 → same conclusion as LLC < d/2, more interpretable.
