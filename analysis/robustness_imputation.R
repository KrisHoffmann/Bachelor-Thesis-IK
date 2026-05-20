# =============================================================================
# robustness_imputation.R
# Phase D — Missingness robustness check (deferred)
#
# NOTE: missingness robustness check requires re-matching on the full
# pre-rule-iii pool. Rule iii was applied at Stage 9 (matching), so the
# stage9_matched_1to3.parquet already excludes the ~132 Δ-comments where
# A_social or A_hypothetical was NaN in stage8. Implementing the
# mean-imputation + missing_index dummy check requires re-running the
# matching pipeline on the full eligible pool before rule iii filtering.
# This is deferred — implement only if supervisor requests it.
# =============================================================================

cat("\n", strrep("=", 70), "\n", sep = "")
cat("robustness_imputation.R — DEFERRED\n")
cat(strrep("=", 70), "\n", sep = "")
cat("\n")
cat("NOTE: missingness robustness check requires re-matching on the full\n")
cat("pre-rule-iii pool. This is deferred — implement only if supervisor\n")
cat("requests it.\n")
cat("\n")
cat("Background:\n")
cat("  Rule iii excluded ~132 delta-comments where A_social or\n")
cat("  A_hypothetical was NaN in the stage8 output. These comments were\n")
cat("  filtered before matching, so they are absent from\n")
cat("  stage9_matched_1to3.parquet. A mean-imputation + missing_index\n")
cat("  dummy check cannot be run post-hoc on the matched sample alone;\n")
cat("  it requires re-running the full matching pipeline on the\n")
cat("  pre-rule-iii eligible pool.\n")
cat("\n")
cat("To implement:\n")
cat("  1. Rebuild stage9 matching without rule iii.\n")
cat("  2. Impute NaN A_social / A_hypothetical with corpus means.\n")
cat("  3. Add binary missing_A_social and missing_A_hypothetical dummies.\n")
cat("  4. Rerun regression_prep.R and models.R on the new matched dataset.\n")
cat("\n")
cat("=== robustness_imputation.R exiting cleanly (no models fitted) ===\n")
