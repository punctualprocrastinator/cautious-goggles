# Run this cell to fix the ValueError in Notebook 3, Cell 6
# Bug: X_test_nat had full 10240 features; concatenation needed top-50 sliced version

# ── Corrected Cell 6 ─────────────────────────────────────────
seq_identities   = np.load('embeddings/sequence_identities.npy')
blast_identities = np.load('embeddings/blast_identities.npy')

# Use the already-scaled top-50 slices (not the full 10240)
X_test_all_sae  = np.concatenate([X_te_sae,                              # (n_test, 50)
                                   sc_sae.transform(rdsg_acts[:, top_feat_idx])])  # (n_rdsg, 50)
X_test_all_full = np.concatenate([X_te_full,                             # (n_test, 10240)
                                   sc_full.transform(rdsg_acts)])         # (n_rdsg, 10240)
y_test_all      = np.concatenate([y_te, np.ones(len(rdsg_acts))])

print(f"X_test_all_sae:  {X_test_all_sae.shape}")
print(f"X_test_all_full: {X_test_all_full.shape}")
print(f"y_test_all:      {y_test_all.shape}")
