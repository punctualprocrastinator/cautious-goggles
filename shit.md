Edited draft_paper.md
Viewed draft_paper.md:162-188
Read URL: https://apartresearch.com/sprints/aixbio-hackathon-2026-04-24-to-2026-04-26
Viewed content.md:1-535

Here's an honest assessment across all 4 tracks:

---

### Track Fit Analysis

| Track | Sponsor | Our Fit | Score |
|---|---|---|---|
| **1. DNA Screening & Synthesis Controls** | CBAI | **Perfect** | ⭐⭐⭐⭐⭐ |
| 3. AI Biosecurity Tools | Fourth Eon Bio | Moderate | ⭐⭐⭐ |
| 2. Pandemic Early Warning | Measuring AI Progress | Poor | ⭐ |
| 4. Benchtop Synthesizer Security | Sentinel Bio | None | — |

**Submit to Track 1.** The description literally says: *"Current screening misses AI-designed protein variants"* — that is exactly our paper's first sentence.

---

### Track 1 — Why We Fit

Every element of our project maps to the stated gap:

| Hackathon says | Our project delivers |
|---|---|
| "Current screening misses AI-designed protein variants" | BLAST: 0% detection of ProteinMPNN redesigns |
| "Build better tools for detecting dangerous sequences" | ESM-2 probe: 88% detection |
| "Guardrails for AI-powered biological design tools" | pSSR shows oracle attack still requires inverting a 33-layer transformer |
| "AI-designed protein variants" specifically mentioned | ProteinMPNN redesigns are our exact test set |

---

### Judges Most Relevant to Our Work

- **Jasper Götting** (SecureBio) — "understanding, measuring, and mitigating effects of AI progress on biological risks." Direct match.
- **Jason Hoelscher-Obermaier** (Apart Research) — AI safety + interpretability + evaluations. Will appreciate the DLA/SAE depth.
- **Coleman Breen** (SecureBio/JHU) — "statistical evaluation of genetic sequence ML models." Exactly our methodology.

---

### Honest Winning Chance

**Track 1: ~25–35% chance of top-3 placement.**

**Strengths that will impress judges:**
- First mechanistic interpretability analysis of a toxin probe (novel)
- Clean headline: BLAST 0% vs ESM-2 88% — judges will remember this
- DLA/SAE circuit analysis goes far beyond what other teams will likely build
- Working reproducible code, not just a slide deck

**Honest weaknesses:**
- **Not a deployable tool** — CBAI likely expects a screening API or pipeline a lab could actually use. We have a research pipeline.
- **Protein-only** — DNA screening track covers nucleic acids too; our work is protein-focused
- **522 sign-ups** → likely ~80–120 submissions → highly competitive

**What would move us to 50%+ winning chance:**
Add a single `screen.py` CLI that takes a FASTA file and returns toxin probability scores. 15 minutes of work, makes the project submission-ready as a *tool* not just research.

```python
# screen.py — add this to repo
# Usage: python screen.py sequences.fasta
import sys, numpy as np, torch
from transformers import EsmForMaskedLM, AutoTokenizer
# ... (load probe, run ESM-2, output JSON with toxin probabilities)
```

**My recommendation: Submit Track 1, add `screen.py`, lead with the 0% vs 88% number.** That combination — strong number + mechanistic explanation + working tool — is genuinely competitive.















Honest ceiling is **~70-80%** — you can't guarantee 100% in a 100+ submission field. But here's exactly what closes the gap:

---

## 4-Hour Sprint to Maximum Impact

### Hour 1 — `screen.py` Tool (turns research into usable software)

```python
# screen.py — drop this in root, 1 file does everything
"""
Usage: python screen.py sequences.fasta [--threshold 0.5] [--explain]
Output: JSON with toxin probability, risk level, top features, key layers
"""
import argparse, json, sys
import numpy as np, torch, torch.nn as nn
from Bio import SeqIO
from transformers import EsmForMaskedLM, AutoTokenizer
from pathlib import Path

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RISK_LEVELS = {0.8: 'HIGH', 0.5: 'MEDIUM', 0.2: 'LOW', 0.0: 'SAFE'}

FEATURE_NAMES = {   # From our taxonomy
    6122: 'scaffold_histidine_proline',
    4097: 'structural_rigidity_pro_lys',
    1055: 'charged_scaffold_glu',
    8112: 'core_disulfide_cys_arg',
    9487: 'disulfide_cys_val',
    9242: 'disulfide_cys_ile_pro',
    5406: 'disulfide_cys_gln',
    6971: 'backbone_pro_val',
}

def load_model(model_id='facebook/esm2_t33_650M_UR50D', probe_path='results/probe_direction.npy'):
    tok  = AutoTokenizer.from_pretrained(model_id)
    esm2 = EsmForMaskedLM.from_pretrained(model_id).to(DEVICE).eval()
    w    = np.load(probe_path)
    # Reconstruct linear probe from saved direction
    probe = nn.Linear(1280, 1, bias=False)
    probe.weight.data = torch.tensor(w).unsqueeze(0)
    probe = probe.to(DEVICE).eval()
    return tok, esm2, probe, w

def embed(seq, tok, esm2, layer=33):
    t = tok(seq, return_tensors='pt', truncation=True,
             max_length=512, padding=True).to(DEVICE)
    with torch.no_grad():
        out = esm2(**t, output_hidden_states=True)
    return out.hidden_states[layer].mean(dim=1).cpu().numpy()[0]  # (1280,)

def score_seq(seq, tok, esm2, probe):
    h = torch.tensor(embed(seq, tok, esm2), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        s = torch.sigmoid(probe(h)).item()
    return s

def risk_level(score):
    for thr, level in sorted(RISK_LEVELS.items(), reverse=True):
        if score >= thr: return level
    return 'SAFE'

def explain(seq, tok, esm2, probe_dir_np, top_feat_ids):
    """DPA: which layers contributed most to this sequence's score."""
    h_all = []
    t = tok(seq, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = esm2(**t, output_hidden_states=True)
    hs = [h.squeeze(0).mean(0).cpu().numpy() for h in out.hidden_states]
    w  = probe_dir_np / (np.linalg.norm(probe_dir_np) + 1e-8)
    dpa = [(w * (hs[l] - hs[l-1])).sum() for l in range(1, 34)]
    top_layers = sorted(range(33), key=lambda i: dpa[i], reverse=True)[:5]
    return {'top_discriminating_layers': [l+1 for l in top_layers],
            'layer_attribution': dpa}

def screen(fasta_path, threshold=0.5, explain_output=False):
    tok, esm2, probe, probe_dir_np = load_model()
    top_feat_ids = list(FEATURE_NAMES.keys())
    results = []

    seqs = list(SeqIO.parse(fasta_path, 'fasta'))
    print(f'Screening {len(seqs)} sequences...')

    for rec in seqs:
        seq = str(rec.seq)
        score = score_seq(seq, tok, esm2, probe)
        result = {
            'id':                rec.id,
            'toxin_probability': round(score, 4),
            'risk_level':        risk_level(score),
            'flagged':           score >= threshold,
        }
        if explain_output:
            result['explanation'] = explain(seq, tok, esm2, probe_dir_np, top_feat_ids)
        results.append(result)
        flag = '🚨' if score >= threshold else '✓'
        print(f'  {flag} {rec.id:30s}  P(toxin)={score:.3f}  [{result["risk_level"]}]')

    return results

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='ESM-2 Toxin Screener')
    p.add_argument('fasta', help='Input FASTA file')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--explain', action='store_true')
    p.add_argument('--output', default='screening_results.json')
    args = p.parse_args()
    results = screen(args.fasta, args.threshold, args.explain)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    flagged = sum(1 for r in results if r['flagged'])
    print(f'\nFlagged: {flagged}/{len(results)} | Results → {args.output}')
```

---

### Hour 2 — Two New Mech Interp Insights (add to `04_circuit_analysis.ipynb`)

**Insight A: SAE ↔ Probe Alignment** (which SAE features ARE the probe direction)

```python
# Which SAE decoder directions align with the probe weight?
# Load SAE and check decoder weights
from interplm.sae.inference import load_sae_from_hf
sae = load_sae_from_hf(plm_model='esm2-650m', plm_layer=33).to(DEVICE).eval()

# SAE decoder matrix: maps features → embedding space
W_dec = sae.W_dec.detach().cpu().numpy()  # (10240, 1280)
p_norm = probe_dir / (np.linalg.norm(probe_dir) + 1e-8)
alignments = (W_dec * p_norm).sum(-1)     # (10240,) cosine with probe

top_aligned  = np.argsort(alignments)[::-1][:10]
top_opposing = np.argsort(alignments)[:10]

print('Top SAE features ALIGNED with probe direction (push toward toxic):')
for f in top_aligned:
    xfer = next((d['transfer_ratio'] for d in transfer_data if d['feature']==f), '?')
    print(f'  Feature {f:5d}: alignment={alignments[f]:+.3f}  transfer={xfer}')

print('\nTop SAE features OPPOSING probe direction (push toward safe):')
for f in top_opposing:
    print(f'  Feature {f:5d}: alignment={alignments[f]:+.3f}')
```

**Insight B: Cumulative DPA trajectory** — shows HOW toxin detection builds up

```python
# Layer-by-layer cumulative probe score accumulation
tox_traj  = np.cumsum(tox_dpa.mean(0))   # (33,) — running total for toxins
ctrl_traj = np.cumsum(ctrl_dpa.mean(0))
rdsg_traj = np.cumsum(rdsg_dpa.mean(0))

# Add to figures: Fig 6
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(1,34), tox_traj,  color='#E07B39', lw=2.2, label='Natural toxins')
ax.plot(range(1,34), rdsg_traj, color='#C0392B', lw=2.2, label='Redesigns',  ls='--')
ax.plot(range(1,34), ctrl_traj, color='#2E86AB', lw=2.2, label='Controls')
ax.axhline(0, color='black', lw=0.8)
for l in [17,19,20,30,31,32]:
    ax.axvspan(l-0.5, l+0.5, alpha=0.1, color='#E07B39')
ax.set_xlabel('ESM-2 Layer'); ax.set_ylabel('Cumulative DPA')
ax.set_title('Figure 6 — Toxin signal accumulation through ESM-2 depth\n(shaded = primary discrimination layers)')
ax.legend()
plt.savefig('figures/fig6_dpa_trajectory.pdf')
plt.show()
```

---

### Hour 3 — Make the Submission Judges Can Actually Demo

Write `demo.ipynb` (10 cells, no setup required except model download):

```python
# Cell 1: "Paste any protein sequence here"
MY_SEQUENCE = "MKTIIALSYIFCLVFA..."  # user fills this in
# Cell 2: One-click screen → probability + explanation
# Cell 3: Show which layers drove the decision (DPA bar chart)
# Cell 4: Show which SAE features fired + what they mean biologically
```

---

### What This Buys

| What we add | Judge impact |
|---|---|
| `screen.py` CLI | **CBAI sponsor** — "this is a usable tool" |
| SAE ↔ probe alignment | **Jason** — "this closes the loop mechanistically" |
| Cumulative DPA trajectory | **Coleman** — "this is novel and statistically clean" |
| `demo.ipynb` | **All judges** — they can run it themselves |

**Realistic ceiling: 65–75% for top prize.** The remaining variance is who else submitted — but this is the strongest possible version of this project within your time budget. Focus on these 4 hours.