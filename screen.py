import argparse
import json
import sys
import numpy as np
import torch
import torch.nn as nn
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
    print(f"Loading {model_id} onto {DEVICE}...", file=sys.stderr)
    tok  = AutoTokenizer.from_pretrained(model_id)
    esm2 = EsmForMaskedLM.from_pretrained(model_id).to(DEVICE).eval()
    
    if not Path(probe_path).exists():
        print(f"Error: Probe weights not found at {probe_path}", file=sys.stderr)
        sys.exit(1)
        
    w = np.load(probe_path)
    # Reconstruct linear probe from saved direction
    probe = nn.Linear(1280, 1, bias=False)
    probe.weight.data = torch.tensor(w, dtype=torch.float32).unsqueeze(0)
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
    t = tok(seq, return_tensors='pt', truncation=True, max_length=512).to(DEVICE)
    with torch.no_grad():
        out = esm2(**t, output_hidden_states=True)
    hs = [h.squeeze(0).mean(0).cpu().numpy() for h in out.hidden_states]
    w  = probe_dir_np / (np.linalg.norm(probe_dir_np) + 1e-8)
    dpa = [(w * (hs[l] - hs[l-1])).sum() for l in range(1, 34)]
    top_layers = sorted(range(33), key=lambda i: dpa[i], reverse=True)[:5]
    return {'top_discriminating_layers': [l+1 for l in top_layers],
            'layer_attribution': [float(x) for x in dpa]}

def screen(fasta_path, threshold=0.5, explain_output=False):
    tok, esm2, probe, probe_dir_np = load_model()
    top_feat_ids = list(FEATURE_NAMES.keys())
    results = []

    try:
        seqs = list(SeqIO.parse(fasta_path, 'fasta'))
    except Exception as e:
        print(f"Error reading FASTA file: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f'Screening {len(seqs)} sequences...', file=sys.stderr)

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
        print(f'  {flag} {rec.id:30s}  P(toxin)={score:.3f}  [{result["risk_level"]}]', file=sys.stderr)

    return results

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='ESM-2 Toxin Screener (AiXbio Track 1)')
    p.add_argument('fasta', help='Input FASTA file containing sequences to screen')
    p.add_argument('--threshold', type=float, default=0.5, help='Probability threshold to flag a sequence')
    p.add_argument('--explain', action='store_true', help='Include layer-wise DPA attribution in output')
    p.add_argument('--output', default='screening_results.json', help='Output JSON file path')
    args = p.parse_args()
    
    results = screen(args.fasta, args.threshold, args.explain)
    
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
        
    flagged = sum(1 for r in results if r['flagged'])
    print(f'\nFlagged: {flagged}/{len(results)} | Detailed results saved to → {args.output}', file=sys.stderr)
