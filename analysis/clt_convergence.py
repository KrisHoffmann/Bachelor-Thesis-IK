"""
clt_convergence.py
==================
CLT Convergence Analysis — uses the FULL per-comment index file (~598k rows).

Outputs
-------
analysis/output/clt_correlations.csv  — 6-row pairwise correlation table
analysis/output/clt_pca.csv           — PCA loadings and variance explained

Run from the repo root:
    python analysis/clt_convergence.py
"""

import pathlib
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = pathlib.Path(__file__).resolve().parent.parent
DATA_FILE  = REPO_ROOT / "final_dataset_creation/dataset_construction/outputs/stage8_comment_full.parquet"
OUTPUT_DIR = REPO_ROOT / "analysis/output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DIMS = ["A_temporal", "A_spatial", "A_social", "A_hypothetical"]
DIM_LABELS = {"A_temporal": "A_T", "A_spatial": "A_S",
              "A_social": "A_Soc", "A_hypothetical": "A_H"}

# ---------------------------------------------------------------------------
# Load data (only the four index columns)
# ---------------------------------------------------------------------------
print("Loading data …")
df = pd.read_parquet(DATA_FILE, columns=DIMS)
print(f"Loaded {len(df):,} rows.\n")

# ============================================================================
# A2 — Percentage defined (print first so sparsity is obvious up-front)
# ============================================================================
print("=" * 70)
print("A2 — PERCENTAGE DEFINED (non-NaN) PER DIMENSION")
print("=" * 70)
for col in DIMS:
    n_def = df[col].notna().sum()
    pct   = 100 * n_def / len(df)
    short = DIM_LABELS[col]
    print(f"  {short:6s}  {n_def:>9,} / {len(df):,}  ({pct:5.1f}% defined)")
print()

# ============================================================================
# A1 — Pairwise Pearson correlation matrix (pairwise-complete observations)
# ============================================================================
print("=" * 70)
print("A1 — PAIRWISE PEARSON CORRELATIONS (pairwise-complete N per pair)")
print("=" * 70)

pairs = []
for i, c1 in enumerate(DIMS):
    for c2 in DIMS[i+1:]:
        mask  = df[c1].notna() & df[c2].notna()
        n     = mask.sum()
        x, y  = df.loc[mask, c1].values, df.loc[mask, c2].values
        r, p  = stats.pearsonr(x, y)
        pairs.append({
            "Pair":  f"{DIM_LABELS[c1]}–{DIM_LABELS[c2]}",
            "col1":  c1,
            "col2":  c2,
            "r":     r,
            "p":     p,
            "N":     n,
        })

corr_df = pd.DataFrame(pairs)

# --- Tidy 6-row summary table ---
print("\nTidy pairwise summary:")
print(f"{'Pair':<14} {'r':>7}  {'p':>8}  {'N':>10}")
print("-" * 45)
for _, row in corr_df.iterrows():
    p_str = f"{row['p']:.3f}" if row["p"] >= 0.0005 else f"{row['p']:.2e}"
    print(f"  {row['Pair']:<12} r = {row['r']:+.2f},  p = {p_str},  N = {int(row['N']):>9,}")

# --- 4×4 matrix display ---
print("\n4×4 correlation matrix (upper triangle = r; lower = N; diagonal = 1.000):")
short_names = [DIM_LABELS[c] for c in DIMS]
# Build lookup: (col1, col2) -> r
r_lookup = {}
n_lookup = {}
for _, row in corr_df.iterrows():
    r_lookup[(row["col1"], row["col2"])] = row["r"]
    r_lookup[(row["col2"], row["col1"])] = row["r"]
    n_lookup[(row["col1"], row["col2"])] = row["N"]
    n_lookup[(row["col2"], row["col1"])] = row["N"]

header = f"{'':>8}" + "".join(f"{s:>12}" for s in short_names)
print(header)
for ri, c1 in enumerate(DIMS):
    row_str = f"  {DIM_LABELS[c1]:<6}"
    for ci, c2 in enumerate(DIMS):
        if c1 == c2:
            row_str += f"{'1.000':>12}"
        elif ci > ri:
            row_str += f"{r_lookup[(c1, c2)]:>+12.3f}"
        else:
            row_str += f"{int(n_lookup[(c1, c2)]):>12,}"
    print(row_str)
print("  (upper triangle = r; lower triangle = N for that pair)")

# Save
save_df = corr_df[["Pair", "r", "p", "N"]].copy()
save_df.to_csv(OUTPUT_DIR / "clt_correlations.csv", index=False, float_format="%.6f")
print(f"\nSaved: {OUTPUT_DIR / 'clt_correlations.csv'}")

# ============================================================================
# A3 — PCA on A_Soc and A_H (the two usable dimensions)
# ============================================================================
print()
print("=" * 70)
print("A3 — PCA ON A_Soc AND A_H (rows where both are non-NaN)")
print("=" * 70)

pca_cols = ["A_social", "A_hypothetical"]
mask_pca = df["A_social"].notna() & df["A_hypothetical"].notna()
df_pca   = df.loc[mask_pca, pca_cols]
print(f"\nRows used for PCA: {len(df_pca):,} "
      f"({100 * len(df_pca) / len(df):.1f}% of full dataset)")

# Standardise
scaler  = StandardScaler()
X_std   = scaler.fit_transform(df_pca.values)

# PCA (2 components; with 2 variables this is the full decomposition)
pca = PCA(n_components=2)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pca.fit(X_std)

loadings   = pca.components_          # shape (2, 2)
var_exp    = pca.explained_variance_ratio_
cum_var    = np.cumsum(var_exp)

print()
for i in range(2):
    l_soc = loadings[i, 0]
    l_hyp = loadings[i, 1]
    print(f"PC{i+1} explains {var_exp[i]*100:.1f}% of variance; "
          f"loadings: A_Soc = {l_soc:+.3f}, A_H = {l_hyp:+.3f}")

print(f"\nCumulative variance explained: PC1 = {cum_var[0]*100:.1f}%, "
      f"PC1+PC2 = {cum_var[1]*100:.1f}%")

# Analytical check: for two standardised variables the 2x2 correlation matrix is
# [[1, r], [r, 1]], whose eigenvalues are (1+r) and (1-r).  PCA orders components
# by *decreasing* eigenvalue, so PC1 always captures the larger one.
# When r > 0: PC1 eigenvalue = 1+r, share = (1+r)/2.
# When r < 0: PC1 eigenvalue = 1-r = 1+|r|, share = (1-r)/2 = (1+|r|)/2.
# In both cases PC1 share = (1 + |r|) / 2, consistent with the correlation above.
r_soc_h = r_lookup[("A_social", "A_hypothetical")]
analytic_pc1 = (1 + abs(r_soc_h)) / 2
print(f"\nAnalytical check — (1 + |r_{{A_Soc,A_H}}|) / 2 = "
      f"(1 + {abs(r_soc_h):.4f}) / 2 = {analytic_pc1:.4f}  "
      f"[PCA reports {var_exp[0]:.4f} — consistent: {np.isclose(analytic_pc1, var_exp[0], atol=1e-4)}]")

print()
print("Thesis-reportable PCA result:")
print(f'  "PC1 explains {var_exp[0]*100:.1f}% of variance '
      f'(loadings: A_Soc = {loadings[0,0]:+.2f}, A_H = {loadings[0,1]:+.2f}); '
      f'PC2 explains the remaining {var_exp[1]*100:.1f}%."')

# Save
pca_rows = []
for i in range(2):
    pca_rows.append({
        "component":           f"PC{i+1}",
        "loading_A_social":    loadings[i, 0],
        "loading_A_hypothetical": loadings[i, 1],
        "variance_explained":  var_exp[i],
        "cumulative_variance": cum_var[i],
    })
pca_out = pd.DataFrame(pca_rows)
pca_out.to_csv(OUTPUT_DIR / "clt_pca.csv", index=False, float_format="%.6f")
print(f"\nSaved: {OUTPUT_DIR / 'clt_pca.csv'}")

print("\n=== clt_convergence.py complete ===")
