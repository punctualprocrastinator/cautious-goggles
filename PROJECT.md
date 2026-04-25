# Geometric Signatures of Functional Evasion
## Do Protein Language Model Probes Fail Predictably at Biosecurity Screening?

> **Track:** DNA Screening & Synthesis Controls — AIxBio Hackathon (Apr 24–26, 2026)  
> **Sponsors:** CBAI (Cambridge Boston Alignment Initiative) · Cambridge Biosecurity Hub · Apart Research  
> **Fellowship deadline:** AIxBiosecurity Research Fellowship — April 27, 23:59 UTC

---

## TL;DR

AI protein design tools (ProteinMPNN) can redesign known toxins — changing ~50% of amino acid residues while preserving 3D structure and function — so that current homology-based DNA synthesis screeners fail to flag them. We ask: do ESM-2 linear probes trained on natural toxins generalize to these AI-redesigned sequences? And can Singular Learning Theory (LLC) predict *when* they fail, before failure occurs?

Both outcomes are publishable. If probes generalize → pLM-based screening works and here is why (mechanistic). If probes fail → we have a geometric signature of the failure that tells the field what architecture is actually needed.

---

## Deadlines

| Date | Milestone |
|------|-----------|
| **Apr 14** | Project start. SLT tooling verified (`canonical-interp`, `devinterp`). |
| **Apr 15** | LLC estimator confirmed correct on closed-form logistic regression. |
| **Apr 16** | UniProt toxin dataset downloaded and clustered (MMseqs2, 30% identity). ESM-2 embedding pipeline running. |
| **Apr 17** | All embeddings extracted (layers 1, 9, 18, 24, 30, 33). Train/test splits finalised. |
| **Apr 18** | AlphaFold structures pulled for 100 toxin representatives. ProteinMPNN redesign jobs launched. |
| **Apr 19** | Probe training complete across all layers. Best layer identified by AUROC. |
| **Apr 20** | LLC computation on trained probe. Generalization curve experiment drafted. |
| **Apr 21** | Full generalization curve complete. BLAST baseline computed. SAE features loaded. |
| **Apr 22** | SAE feature overlap analysis complete. UMAP generated. All four figures drafted. |
| **Apr 23** | End-to-end pipeline rehearsal. `reproduce.sh` script written. Code pushed to GitHub. |
| **Apr 24** | Hackathon Day 1: Final experimental runs with locked hyperparameters. |
| **Apr 25** | Hackathon Day 2: Research report written. |
| **Apr 26** | Hackathon Day 3: Submit project. |
| **Apr 27** | Apply to AIxBiosecurity Research Fellowship (this project IS the application). |

---

## Background and Motivation

### The Screening Gap

DNA synthesis companies screen orders for dangerous sequences by comparing them to databases of known "sequences of concern" — a homology-based approach. This has worked well against natural pathogens. It is failing against AI.

In October 2025, Wittmann et al. (*Science*) used ProteinMPNN, EvoDiff-MSA, and EvoDiff-Seq to redesign 72 dangerous proteins into >76,000 variants. Most evaded all four major screening tools. Patches were deployed, but the March 2026 bioRxiv follow-up confirmed that fragment-level evasion remains an open problem.

The proposed fix is *function-based screening* — detecting dangerous proteins by what they *do*, not by sequence similarity to known threats. Protein language models (pLMs) like ESM-2 are the obvious candidate technology, since they are known to encode functional information in their internal representations. But nobody has tested whether this functional encoding survives AI-driven sequence redesign.

### The Interpretability Question

This is not just a biosecurity engineering problem. It is a mechanistic question about what ESM-2 actually learns.

Does ESM-2 encode *dangerous function* (toxin activity, virulence mechanism) as a robust, geometry-stable feature in its representation space — one that persists even when the sequence is scrambled by ProteinMPNN? Or does it encode *dangerous sequence patterns* that collapse under redesign?

Singular Learning Theory (SLT) gives us a principled tool to characterise this geometrically: the Local Learning Coefficient (LLC) measures the effective dimensionality of a model's solution in weight space. A probe with high LLC exploits a complex, high-dimensional region of weight space. A probe with low LLC has found a degenerate solution — flat, broad, reliant on fewer directions. We hypothesise that the probe's LLC, computed at its trained weights, predicts whether it will generalise to redesigned sequences.

---

## Research Questions

1. **Detection:** Do logistic regression probes trained on ESM-2 embeddings of natural toxins generalise to AI-redesigned sequences with <40% sequence identity to training data?
2. **Geometry:** Does the LLC of the trained probe predict the generalization gap? Is there a phase transition in probe performance as sequence identity to training data drops?
3. **Mechanism:** Which SAE features in ESM-2 activate on both natural toxins and AI-redesigns (function features) vs. only on natural toxins (sequence features)?
4. **Baseline:** How does probe AUROC on redesigned sequences compare to BLAST-based screening sensitivity on the same set?

---

## Methods

### Data

**Positive class — Toxin sequences**
- Source: UniProtKB/Swiss-Prot, keyword filter `Toxin` + `Venom protein`
- Expected size: ~8,000 curated sequences
- Clustering: MMseqs2 at 30% identity → ~1,500 representative sequences
- Split: 80/20 train/test, stratified by cluster

**Negative class — Non-toxic controls**
- Source: UniProtKB/Swiss-Prot, excluding all toxin/virulence GO terms
- Matched by length distribution to positive class
- Same clustering and split procedure

**AI-redesigned sequences (the evasion test set)**
- Source: AlphaFold DB structures for 100 toxin representatives (EBI REST API)
- Redesign: ProteinMPNN, 10 sequences per structure = 1,000 redesigns
- Filter: keep only redesigns with <40% identity to any Swiss-Prot toxin (BioPython pairwise)
- These sequences would pass BLAST-based screening — they are the attack set

### Models

**Protein embeddings**
- ESM-2 650M (`facebook/esm2_t33_650M_UR50D`)
- Layers extracted: 1, 9, 18, 24, 30, 33 (InterPLM checkpoint layers)
- Pooling: mean over sequence length → 1280-dim vector per protein

**Probes**
- Logistic regression (scikit-learn) trained per layer
- Bayesian logistic regression (PyTorch + Laplace approximation) for WAIC computation
- Training: natural toxins vs non-toxic controls only
- Evaluation: held-out natural toxins AND AI-redesigned sequences (no retraining)

**Baseline**
- BLAST against Swiss-Prot toxin sequences (NCBI BLAST API)
- Report sensitivity at 30% identity threshold (standard screening cutoff)

**SAE features**
- Pre-trained InterPLM SAE weights for ESM-2 650M layer 18: `load_sae_from_hf("esm2-650m", 18)`
- Dictionary size: 10,240 features
- Analysis: precision/recall of each feature against toxin annotation on natural vs redesigned sequences
- *Methodological note:* We use linear probes for toxicity classification and SAEs for mechanistic discovery — consistent with the emerging consensus that SAEs are discovery tools, not detection tools (see Templeton et al., arXiv:2506.23845).

**LLC computation**
- Library: `canonical-interp` (pip install)
- Applied to: trained logistic regression probe weights (not ESM-2 — tractable)
- SGLD hyperparameters: β=0.1, γ=100, 2,000 steps (verify on toy closed-form first)
- Computed at probe weights trained on full natural toxin training set

### Key Experiment: The Generalization Curve

Train probe once on natural toxins. Then evaluate it on test sequences grouped by their maximum sequence identity to any training sequence. Identity bins: [90-100%, 70-90%, 50-70%, 30-50%, 10-30%, <10%].

For each bin, compute:
- Probe AUROC
- BLAST sensitivity (same sequences)
- Probe cross-entropy loss
- WAIC of the Bayesian probe

Plot all four against identity. This is the main result.

**Hypothesis:** AUROC and BLAST sensitivity diverge below 40% identity, with the probe maintaining higher detection than BLAST. LLC, computed once at training weights, predicts the identity threshold at which probe performance begins to degrade via the theoretical generalization bound from SLT: expected error ~ λ·log(n)/n.

---

## Experiments

### Experiment 1 — Layer Selection (Day 4)
Train probes at all six ESM-2 650M layers. Report AUROC on held-out natural toxin test set. Identify best layer.

**Expected result:** Peak AUROC at layer 18 or 24 (mid-to-late layers encode strongest functional signal per prior work). AUROC > 0.90 on natural toxins.

**Fallback:** If AUROC < 0.75 on any layer, switch to ESM-2 8M with InterPLM 8M SAEs (faster, more conservative). Result is still valid.

### Experiment 2 — Generalization Curve (Days 7–8)
Main experiment. Evaluate probe across identity bins. Compare to BLAST.

**Expected result (optimistic):** Probe maintains AUROC > 0.80 down to 40% identity. BLAST sensitivity drops to ~30% below 40% identity. The gap at <40% identity is the contribution.

**Expected result (pessimistic):** Probe AUROC also drops sharply below 50% identity. This is still publishable: we characterise *where* pLM-based screening fails and provide the LLC as a predictive signature of that failure.

**Worst case mitigation:** If both probe and BLAST fail equally, the SAE feature analysis (Experiment 4) becomes the main result — which features survive redesign, and why.

### Experiment 3 — LLC Computation (Day 7)
Compute LLC at the trained probe weights. Report: LLC value, confidence interval across SGLD chains, sensitivity to hyperparameters (run 3 settings).

**Expected result:** LLC < d/2 where d=1280 (probe input dimension), confirming the probe has found a degenerate/sparse solution — consistent with ESM-2 encoding toxin function in a low-dimensional subspace. If true, this is a geometric explanation for why the probe generalises: the function is encoded in few directions, not all 1280.

**Connection to generalization curve:** Use SLT asymptotic: expected generalization error ≈ λ·log(n)/n. Fit λ from the generalization curve. Does the fitted λ match the LLC estimate? If yes, SLT gives a predictive theory for when screening fails.

### Experiment 4 — SAE Feature Analysis (Days 8–9)
Extract InterPLM SAE feature activations for:
- (A) Natural toxins (training set)
- (B) AI-redesigned sequences (evasion set)
- (C) Non-toxic controls

For each of the top 200 features active on (A), compute activation on (B) and (C). Classify features as:
- **Function features:** high activation on both (A) and (B), low on (C) → these encode dangerous *function*, survive redesign
- **Sequence features:** high activation on (A), low on (B) and (C) → these encode dangerous *sequence patterns*, fail under redesign
- **Noise features:** low or random activation across all groups

**Expected result:** A small set (20–50) of function features survive redesign. These are the interpretable basis for a better screener. The sequence features explain why current screening fails.

### Experiment 5 — UMAP (Day 9)
2D UMAP of ESM-2 layer-18 embeddings, coloured by:
- Natural toxins (orange)
- AI-redesigned sequences (red)
- Non-toxic controls (blue)

**Expected result:** Two scenarios, both interesting:
- (A) Redesigns cluster with natural toxins → function is geometrically preserved. Probes should work.
- (B) Redesigns drift toward controls → function is NOT preserved in ESM-2 space. Probes cannot work.

The UMAP result predicts the generalization curve result and provides the visual for the paper.

---

## Expected Figures

### Figure 1 — Layer probe AUROC on natural toxins
Bar chart, x-axis = ESM-2 layer, y-axis = AUROC on held-out natural toxin test set. Shows best layer and how quickly performance builds up through layers.

*Audience signal (CBH):* ESM-2 can classify dangerous proteins with high accuracy on natural sequences.
*Audience signal (CBAI):* Functional information peaks in mid-to-late layers, consistent with prior mechanistic interpretability work on pLMs.

### Figure 2 — The generalization curve (MAIN RESULT)
X-axis: max sequence identity to training toxins (binned). Y-axis (left): AUROC (probe) vs sensitivity (BLAST). Y-axis (right): WAIC. LLC value annotated as a horizontal reference line.

This is Figure 1 in the submitted report (biosecurity finding first for CBH judges).

### Figure 3 — SAE feature overlap heatmap
Rows = top 50 SAE features active on natural toxins. Columns = natural toxins / AI-redesigns / non-toxic controls. Colour = mean normalised activation. Annotate function features vs sequence features.

### Figure 4 — UMAP
Scatterplot of ESM-2 layer-18 embeddings. Three point clouds coloured orange/red/blue. Probe decision boundary overlaid as a contour.

---

## Expected Paper Contribution

**For biosecurity venues (CBH framing):**
> We benchmark ESM-2 linear probes against BLAST-based screening on AI-redesigned toxin sequences and characterise the identity threshold at which each approach fails. We provide the first benchmark dataset of ProteinMPNN-redesigned toxins for biosecurity screening evaluation, and identify the ESM-2 SAE features that encode dangerous function robustly vs those that encode surface sequence patterns.

**For ML/interpretability venues (CBAI framing):**
> We apply Singular Learning Theory to characterise the loss landscape geometry of protein function probes trained on ESM-2 representations. We show that the LLC of a trained toxicity probe predicts its out-of-distribution generalization as sequences diverge from the training distribution, and that ESM-2 SAE features decompose cleanly into function-preserving and sequence-dependent directions.

**Target venue:** NeurIPS Workshop on ML for Structural Biology (2026) or AI for Science Workshop. Expandable to full NeurIPS main track paper with wet-lab validation of function preservation (requires biology collaborators — Nicole Wheeler at University of Birmingham is the natural contact via CBH).

---

## Related Work

### Core Biosecurity Papers (Direct Motivation)

**Wittmann et al. (Science, Oct 2025)**
"Strengthening nucleic acid biosecurity screening against generative protein design tools"
— The paper this project directly extends. Used ProteinMPNN, EvoDiff-MSA, EvoDiff-Seq to redesign 72 dangerous proteins into >76,000 variants. Most evaded all four major screening tools. Proposed function-based screening as the fix. Did not test pLM probes. Did not use SLT.
*Role in our paper:* Primary motivation. We implement their attack (ProteinMPNN redesign) and test whether pLM probes solve the problem they identified.

**Wittmann et al. (bioRxiv, March 2026)**
"The Limits of Sequence-Based Biosecurity Screening Tools in the Age of AI-Assisted Protein Design"
— Follow-up testing fragment-level evasion. Found that patched tools detect fragments as short as 50 nucleotides but called for "alternate BSS approaches." Used safe proxy sequences (controlled venoms) — the same approach we use to avoid information hazards.
*Role in our paper:* Provides the safe proxy methodology we follow. Directly motivates our work.

**Esvelt et al. (Nature Communications, Jan 2026)**
"Assembling unregulated DNA segments bypasses synthesis screening: regulate fragments as select agents"
— Demonstrated that unregulated DNA fragments can be assembled into dangerous sequences (1918 influenza) from 38 providers. Proved that sequence-level screening is insufficient regardless of accuracy.
*Role in our paper:* Background context. Motivates function-based approaches.

**The Defending Against Splitting paper (bioRxiv, March 2025)**
Developed the Gene Edit Distance algorithm for split-order detection.
*Role in our paper:* Related problem, different approach. We focus on function-level not fragment-level.

### Core ML Papers (Methodological Foundation)

**Simon & Zou — InterPLM (Nature Methods, Oct 2025; preprint Nov 2024)**
"Discovering Interpretable Features in Protein Language Models via Sparse Autoencoders"
— Trained SAEs on ESM-2 8M and 650M. Identified 2,548 interpretable features per layer correlated with 143 known biological concepts. Released pre-trained SAE weights on HuggingFace (`Elana/InterPLM-esm2-650m`). We use their weights directly.
*Role in our paper:* We use their SAE weights and InterProt visualizer. We extend their annotation-matching framework to toxicity/virulence annotations not in their original paper.

**Adams et al. (Columbia/Ginkgo, bioRxiv Feb 2025; PMC)**
"From Mechanistic Interpretability to Mechanistic Biology: Training, Evaluating, and Interpreting Sparse Autoencoders on Protein Language Models"
— Trained SAEs on ESM-2 residual stream. Found that mid-layer SAEs contain family-specific features. Showed linear probes on SAE latents identify known sequence determinants of thermostability and subcellular localisation.
*Role in our paper:* Validates the SAE-on-pLM approach. Their finding that SAE features predict thermostability supports our hypothesis that they could predict toxicity.

**Gujral et al. (MIT CSAIL, PNAS Aug 2025)**
"Sparse autoencoders uncover biologically interpretable features in protein language model representations"
— Applied SAEs and transcoders to ESM-2. Used Claude to interpret features against GO/UniProt annotations. First published in PNAS — legitimises the methodology for high-impact venues.
*Role in our paper:* Confirms methodology is PNAS-level work. Their UniProt-based annotation matching is what we adapt for toxin annotations.

**Murakami et al. (bioRxiv, Dec 2025)**
"Mechanistic Interpretability of Fine-Tuned Protein Language Models for Nanobody Thermostability Prediction"
— Applied SAEs to fine-tuned ESM-2 for thermostability. Found SAEs decompose dense embeddings into sparse, interpretable features without loss of predictive accuracy.
*Role in our paper:* Confirms SAE features generalise to downstream property prediction tasks — supports our approach.

**DeepMind (Alignment Forum/Medium, March 2025)**
"Negative Results for SAEs on Downstream Tasks"
— Found SAE-reconstructed activations cause 10–40% performance degradation. Shifted to "pragmatic interpretability." Found linear probes often outperform SAEs for detection tasks.
*Role in our paper:* Directly addressed. We use linear probes as primary detection method (GDM-validated), SAEs for interpretability only. The two answer different questions.

### Singular Learning Theory (Theoretical Foundation)

**Watanabe (2009)**
*Algebraic Geometry and Statistical Learning Theory* — Cambridge University Press
The foundational text. Proves that the RLCT λ controls the generalization error of singular models via free energy: F_n = nL̄ + λ log n − (m−1) log log n + O(1).

**Lau et al. (arXiv 2308.12108, updated Sep 2024)**
"The Local Learning Coefficient: A Singularity-Aware Complexity Measure"
— Introduces the LLC as a practical estimator of RLCT at arbitrary loss minima via SGLD. Validates against known closed-form RLCTs. The paper behind the `devinterp` and `canonical-interp` libraries.
*Role in our paper:* Primary SLT methodology. We apply their estimator to the logistic regression probe.

**Furman & Lau (arXiv 2402.03698, 2024)**
"Estimating the Local Learning Coefficient at Scale"
— Extends LLC estimation to deep linear networks up to 100M parameters. Confirms the estimator is accurate and scale-invariant.
*Role in our paper:* Validates that LLC can be estimated accurately for our probe size.

**Hoogland et al. (2024)**
"Phase transitions in the formation of in-context learning in transformer language models"
— Used LLC to detect phase transitions during transformer training. Direct precedent for using LLC to detect qualitative changes in model behaviour.
*Role in our paper:* Most direct precedent. We use LLC to detect phase transitions in probe generalisation as a function of sequence identity — same analytical approach, new domain.

**`canonical-interp` library (PyPI v0.1.2)**
"Efficient tooling for Developmental Interpretability"
— Rewrite of the Timaeus devinterp library. SGLD-based LLC estimator with vmapped GPU parallelism.
*Role in our paper:* Practical implementation. `pip install canonical-interp`.

### Biosecurity SAE Unlearning (Direct Parallel)

**Deeb et al. — CRISP (arXiv:2508.13650, 2025)**
"CRISP: Concept Removal via Inference-time Suppression of Pruned SAE features"
— Parameter-efficient persistent concept unlearning using SAEs on Llama-3.1-8B and Gemma-2-2B, evaluated on the WMDP biosecurity benchmark. Automatically identifies salient SAE features across multiple layers and suppresses their activations without fine-tuning. Two findings directly motivate our work: (1) it confirms that biosecurity SAE features are real and coherent — target features consistently capture viral pathogens, disease transmission mechanisms, and biological threat vectors; (2) it documents that some SAE features encode *entangled* concepts (e.g., overlapping harm-related themes), which is the text-LLM analog of our core research question about protein SAE features (does a feature encode sequence or function, or both?). CRISP operates on text LLMs, not protein LMs — our project asks the protein-space version of the same question.
*Role in our paper:* Primary parallel in related work. Validates the premise that dangerous biological concepts map to identifiable SAE features; frames our entanglement question as the protein-space analog of their text finding. Also provides the WMDP-bio benchmark as a potential future cross-modal evaluation surface.

**Templeton et al. — "Use Sparse Autoencoders to Discover Unknown Concepts, Not to Act on Known Concepts" (arXiv:2506.23845, 2025)**
— Systematic review of SAE performance across task types. Key finding: SAEs show *negative* results when concepts are inputs (detection of prespecified categories such as "does this mention toxins?") but *positive* results when concepts are outputs (discovery tasks — identifying what the model represents without prior specification). SAE-based detection underperforms simple linear probes on prespecified concept classification.
*Role in our paper:* Direct methodology justification. We use linear probes for toxicity classification (concept-as-input detection task, where probes outperform SAEs per this paper) and SAEs for mechanistic discovery of which features survive redesign (concept-as-output discovery task, where SAEs excel). This two-track design directly addresses the failure mode this paper documents. Cited explicitly in Methods.

---

### CBAI Prior Work (Judge Alignment)

**Wright & Sharkey — Addressing Feature Suppression in SAEs (CBAI)**
— Studied how SAEs fail to reconstruct activations perfectly, increasing perplexity. The feature suppression framing directly relates to our question: are toxin function features being suppressed in ESM-2 representations of redesigned sequences?

**Wang et al. — Forbidden Facts (CBAI, arXiv 2312.08793)**
— Studied how Llama-2 "knows" suppressed facts but fails to express them. The analogy to our work: ESM-2 may "know" a sequence is functionally dangerous but fail to express that knowledge in a linearly accessible way for redesigned sequences.

**Wu & Hilton — Estimating Probabilities of Rare Outputs (CBAI)**
— Addresses low-probability estimation in language models using importance sampling. The connection: detecting rare dangerous sequences is exactly a low-probability estimation problem in the synthesis order distribution.

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| ProteinMPNN too slow on CPU | Medium | Use Colab T4 GPU (free tier) or reduce to 50 structures |
| Too few redesigns pass <40% identity filter | Medium | Lower threshold to 50% — still below BLAST screening cutoffs |
| ESM-2 650M OOM on available GPU | Low | Fall back to ESM-2 150M; results still valid |
| LLC estimate unstable across chains | Medium | Run 5 chains, report mean ± std. If variance too high, use WAIC as primary SLT metric |
| Probe AUROC low on natural toxins (<0.80) | Low | Switch to ESM-2 fine-tuned on toxin classification (HuggingFace models available) |
| Interesting finding already published | Low | No paper has tested this on AI-redesigned sequences with LLC |

---

## Repo Structure (Target)

```
project/
├── README.md
├── reproduce.sh          # Single script to run full pipeline
├── data/
│   ├── download_uniprot.py
│   ├── cluster_sequences.py
│   └── pull_alphafold.py
├── embeddings/
│   ├── extract_esm2.py
│   └── embed_redesigns.py
├── probes/
│   ├── train_probe.py
│   ├── eval_generalization.py
│   └── bayesian_probe.py
├── slt/
│   ├── compute_llc.py
│   └── generalization_curve.py
├── sae/
│   ├── load_interplm.py
│   └── feature_analysis.py
├── redesign/
│   └── run_proteinmpnn.py
├── figures/
│   ├── fig1_layer_auroc.py
│   ├── fig2_generalization_curve.py
│   ├── fig3_sae_heatmap.py
│   └── fig4_umap.py
└── report/
    └── research_report.md
```

---

## Key Contacts

| Person | Role | Connection |
|--------|------|------------|
| Dr. Nicole Wheeler | University of Birmingham | Co-author Wittmann et al.; CBH symposium speaker. Natural contact for wet-lab validation and IBBIS data access. Email: IBBIS tiered access form. |
| Tessa Alexanian | IBBIS | Co-author Wittmann et al. Contact for safe proxy dataset access. |
| Elana Simon | Stanford / InterPLM | Author of InterPLM. May be contactable for ESM-2 650M SAE questions. |
| CBAI team | emre@cbai.ai | Track sponsor. Fellowship fast-track prize. |
| CBH team | cambiohub.org/contact | Co-organiser. AIxBiosecurity Fellowship (deadline Apr 27). |

---

## The One-Sentence Pitch

*We use Singular Learning Theory to show that ESM-2 probes trained on natural toxins have a measurable geometric signature — the Local Learning Coefficient — that predicts whether they will detect AI-redesigned dangerous proteins, giving biosecurity researchers their first principled tool for knowing when function-based screening will and won't work.*

---

*Last updated: April 24, 2026*
