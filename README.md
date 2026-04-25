# AiXbio Hackathon 2026: Toxin Circuits in ESM-2

**Track 1 Submission: DNA Screening & Synthesis Controls (sponsored by CBAI)**

This repository contains the code and artifacts for our mechanistic interpretability analysis of ESM-2's toxin detection capabilities against ProteinMPNN adversarial redesigns.

## 🏆 Key Findings

- **Sequence screening fails**: ProteinMPNN redesigns evade BLAST screening at every threshold (**0% detection**).
- **Structure-aware models succeed**: A linear probe on ESM-2 detects **88.3%** of the same redesigns, zero-shot.
- **Mechanistic proof**: Using Sparse Autoencoders (SAEs) and Direct Logit Attribution (DLA), we proved that ESM-2 relies on robust structural features that ProteinMPNN cannot remove without destroying the toxin's fold.

## 🚀 Quick Start (For Judges)

The fastest way to see our project in action is to run the interactive demo:

1. Open `demo.ipynb` in your Jupyter environment.
2. Run the cells in order. The first cell installs all necessary dependencies.
3. Paste any protein sequence to screen it and visualize the exact ESM-2 layers responsible for its classification.

## 🛠️ Software Artifacts

To translate our research into actionable biosecurity infrastructure, we have provided a deployable CLI tool:

### `screen.py`
A standalone, zero-shot screening script that analyzes FASTA sequences.

**Usage:**
```bash
# Install dependencies
pip install torch transformers biopython scikit-learn matplotlib numpy

# Screen a FASTA file
python screen.py data/toxins_clustered.fasta --threshold 0.5

# Screen with mechanistic explanations (Layer Attribution)
python screen.py redesign/outputs/redesigns_sample.fa --explain
```

## 📁 Repository Structure

- `screen.py` — The deployable screening CLI tool.
- `demo.ipynb` — Interactive demo notebook.
- `draft_paper.pdf` / `draft_paper.md` — The complete research report with all figures and mechanistic explanations.
- `00_setup.ipynb` to `06_mech_interp_insights.ipynb` — The reproducible research pipeline used to train the probe, patch the residual stream, and generate the figures.
- `data/` — Training sets (natural toxins and controls).
- `redesign/` — ProteinMPNN adversarial redesign outputs.
- `results/` — Pre-computed probe weights, DPA layers, and SAE transfer metrics.
- `figures/` — Generated plots.
