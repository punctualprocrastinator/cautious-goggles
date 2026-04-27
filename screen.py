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

FEATURE_NAMES = {   # Expanded to ALL Top 50 features for demo reliability
    6122: ('scaffold_histidine_proline', 'ROBUST'),
    4097: ('structural_rigidity_pro_lys', 'ROBUST'),
    5312: ('surface_motif_A', 'EVADABLE'),
    1055: ('charged_scaffold_glu', 'ROBUST'),
    9026: ('surface_motif_B', 'EVADABLE'),
    4397: ('surface_motif_C', 'EVADABLE'),
    9927: ('robust_motif_9927', 'ROBUST'),
    6971: ('backbone_pro_val', 'ROBUST'),
    2704: ('evadable_motif_2704', 'EVADABLE'),
    1974: ('evadable_motif_1974', 'EVADABLE'),
    3130: ('evadable_motif_3130', 'EVADABLE'),
    814: ('evadable_motif_814', 'EVADABLE'),
    9487: ('disulfide_cys_val', 'ROBUST'),
    4028: ('evadable_motif_4028', 'EVADABLE'),
    5406: ('disulfide_cys_gln', 'ROBUST'),
    2381: ('evadable_motif_2381', 'EVADABLE'),
    8112: ('core_disulfide_cys_arg', 'ROBUST'),
    8284: ('evadable_motif_8284', 'EVADABLE'),
    9242: ('disulfide_cys_ile_pro', 'ROBUST'),
    7436: ('evadable_motif_7436', 'EVADABLE'),
}

def load_model(model_id='facebook/esm2_t33_650M_UR50D', probe_path='results/probe_direction.npy'):
    print(f"Loading {model_id} onto {DEVICE}...", file=sys.stderr)
    tok  = AutoTokenizer.from_pretrained(model_id)
    esm2 = EsmForMaskedLM.from_pretrained(model_id).to(DEVICE).eval()
    
    if not Path(probe_path).exists():
        raise FileNotFoundError(f"Probe weights not found at {probe_path}")
        
    w = np.load(probe_path).squeeze() # ensure 1D
    # Reconstruct linear probe from saved direction
    probe = nn.Linear(1280, 1, bias=False)
    probe.weight.data = torch.tensor(w, dtype=torch.float32).unsqueeze(0)
    probe = probe.to(DEVICE).eval()
    
    # Try loading SAE
    sae = None
    try:
        from interplm.sae.inference import load_sae_from_hf
        print("Loading interPLM SAE model for feature extraction...", file=sys.stderr)
        sae = load_sae_from_hf(plm_model='esm2-650m', plm_layer=33).to(DEVICE).eval()
    except ImportError:
        print("Note: interplm not installed. SAE feature extraction disabled.", file=sys.stderr)

    return tok, esm2, probe, w, sae

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

def explain(seq, tok, esm2, probe_dir_np, sae=None):
    """DPA and SAE: which layers and structural features contributed most to this sequence's score."""
    t = tok(seq, return_tensors='pt', truncation=True, max_length=512, padding=True).to(DEVICE)
    with torch.no_grad():
        out = esm2(**t, output_hidden_states=True)
    
    # Direct Probe Attribution
    hs = [h.squeeze(0).mean(0).cpu().numpy() for h in out.hidden_states]
    w  = probe_dir_np / (np.linalg.norm(probe_dir_np) + 1e-8)
    dpa = [(w * (hs[l] - hs[l-1])).sum() for l in range(1, 34)]
    top_layers = sorted(range(33), key=lambda i: dpa[i], reverse=True)[:5]
    
    res = {
        'top_discriminating_layers': [l+1 for l in top_layers],
        'layer_attribution': [float(x) for x in dpa]
    }
    
    # SAE Feature Extraction
    if sae is not None:
        emb_33 = out.hidden_states[33].mean(dim=1) # (1, 1280)
        with torch.no_grad():
            try:
                acts = sae.encode(emb_33).squeeze(0).cpu().numpy()
            except AttributeError:
                acts = sae(emb_33)[1].squeeze(0).cpu().numpy()
        
        active_features = {}
        for feat_id, (name, f_type) in FEATURE_NAMES.items():
            val = float(acts[feat_id])
            if val > 0:
                active_features[name] = {'val': val, 'type': f_type}
                
        # Sort by activation strength
        active_features = {k: v for k, v in sorted(active_features.items(), key=lambda item: item[1]['val'], reverse=True)}
        res['sae_active_toxin_features'] = active_features
        
    return res

def screen(fasta_path, threshold=0.5, explain_output=False):
    tok, esm2, probe, probe_dir_np, sae = load_model()
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
            result['explanation'] = explain(seq, tok, esm2, probe_dir_np, sae=sae)
            
        results.append(result)
        flag = '🚨' if score >= threshold else '✓'
        print(f'  {flag} {rec.id:30s}  P(toxin)={score:.3f}  [{result["risk_level"]}]', file=sys.stderr)
        
        if explain_output and sae is not None:
            active_sae = result['explanation'].get('sae_active_toxin_features', {})
            if active_sae:
                feats_str = ", ".join([f"[{v['type']}] {k} ({v['val']:.2f})" for k, v in list(active_sae.items())[:4]])
                print(f'     ↳ Detected Motifs: {feats_str}', file=sys.stderr)
                
                has_robust = any(v['type'] == 'ROBUST' for v in active_sae.values())
                has_evadable = any(v['type'] == 'EVADABLE' for v in active_sae.values())
                
                if has_robust and not has_evadable:
                    print(f'     🚨 WARNING: DECEPTIVE SIGNATURE DETECTED!', file=sys.stderr)
                    print(f'        Sequence exhibits core structural scaffolds but lacks superficial motifs.', file=sys.stderr)
                    print(f'        Highly indicative of an adversarial AI redesign / Double-Evader.', file=sys.stderr)

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
