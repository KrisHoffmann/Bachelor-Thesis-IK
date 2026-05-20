# =============================================================================
# regression_prep.R
# Phase D — Regression preparation for matched case-control sample (1:3)
#
# Run this script FIRST. Output: analysis/df3_prepped.rds + analysis/jb_diagnostics.csv
# =============================================================================

# ── Packages ─────────────────────────────────────────────────────────────────
if (!requireNamespace("arrow",   quietly = TRUE)) install.packages("arrow")
if (!requireNamespace("tseries", quietly = TRUE)) install.packages("tseries")
if (!requireNamespace("dplyr",   quietly = TRUE)) install.packages("dplyr")

library(arrow)
library(tseries)
library(dplyr)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR <- "C:/Users/kris/Documents/Thesis_R"

# ── Load data ─────────────────────────────────────────────────────────────────
cat("\n--- Loading stage9_matched_1to3.parquet ---\n")
df3 <- arrow::read_parquet(file.path(DATA_DIR, "stage9_matched_1to3.parquet"))

# ── Assertions ────────────────────────────────────────────────────────────────
cat("\n--- Running assertions ---\n")

if (nrow(df3) != 23096) {
  stop(sprintf("ASSERTION FAILED: expected 23,096 rows, got %d", nrow(df3)))
}
cat("OK  nrow == 23,096\n")

n_cases <- sum(df3$is_case, na.rm = TRUE)
if (n_cases != 5849) {
  stop(sprintf("ASSERTION FAILED: expected 5,849 cases (is_case == TRUE), got %d", n_cases))
}
cat("OK  sum(is_case) == 5,849\n")

if (anyNA(df3$A_social)) {
  stop("ASSERTION FAILED: A_social contains NULL/NA values")
}
cat("OK  A_social: no nulls\n")

if (anyNA(df3$A_hypothetical)) {
  stop("ASSERTION FAILED: A_hypothetical contains NULL/NA values")
}
cat("OK  A_hypothetical: no nulls\n")

if (!"stratum_id" %in% names(df3)) {
  stop("ASSERTION FAILED: column 'stratum_id' not present in dataset")
}
cat("OK  stratum_id column present\n")

if (!"thread_id" %in% names(df3)) {
  stop("ASSERTION FAILED: column 'thread_id' not present in dataset")
}
cat("OK  thread_id column present\n")

# ── Continuous controls ───────────────────────────────────────────────────────
CONTROLS <- c(
  "noun_to_verb_ratio",
  "type_token_ratio",
  "flesch_kincaid",
  "hedge_density",
  "mean_sentence_length",
  "mean_word_length",
  "punctuation_density",
  "paragraph_count",
  "num_sentences",
  "num_tokens",
  "num_word_tokens",
  "num_urls"
)

# Verify all control columns exist
missing_cols <- setdiff(CONTROLS, names(df3))
if (length(missing_cols) > 0) {
  stop(sprintf("ASSERTION FAILED: missing control columns: %s",
               paste(missing_cols, collapse = ", ")))
}
cat("OK  all control columns present\n")

# ── Impute noun_to_verb_ratio (zero-verb comments have NA) ───────────────────
cat("\n--- Imputing noun_to_verb_ratio ---\n")
n_null_ntvr <- sum(is.na(df3$noun_to_verb_ratio))
cat(sprintf("  NAs in matched sample: %d\n", n_null_ntvr))

if (n_null_ntvr > 0) {
  corpus_mean_ntvr <- mean(df3$noun_to_verb_ratio, na.rm = TRUE)
  cat(sprintf("  Imputing with corpus mean (non-null): %.4f\n", corpus_mean_ntvr))
  df3$noun_to_verb_ratio <- ifelse(
    is.na(df3$noun_to_verb_ratio),
    corpus_mean_ntvr,
    df3$noun_to_verb_ratio
  )
  cat(sprintf("  NAs remaining after imputation: %d\n", sum(is.na(df3$noun_to_verb_ratio))))
} else {
  cat("  No NAs found — no imputation needed\n")
}

# ── Jarque-Bera normality tests ───────────────────────────────────────────────
cat("\n--- Jarque-Bera normality tests (post-imputation, pre-transform) ---\n")

jb_results <- lapply(CONTROLS, function(var) {
  x <- df3[[var]]
  # JB requires finite values; drop any remaining NA/Inf just in case
  x_clean <- x[is.finite(x)]
  jb <- tseries::jarque.bera.test(x_clean)
  data.frame(
    variable  = var,
    JB_stat   = round(jb$statistic, 3),
    p_value   = signif(jb$p.value, 4),
    reject_H0 = jb$p.value < 0.05,
    stringsAsFactors = FALSE,
    row.names = NULL
  )
})

jb_table <- do.call(rbind, jb_results)
rownames(jb_table) <- NULL

cat("\nJarque-Bera results:\n")
print(jb_table, row.names = FALSE)

# Save JB diagnostics
write.csv(jb_table, file = "analysis/jb_diagnostics.csv", row.names = FALSE)
cat("\nSaved: analysis/jb_diagnostics.csv\n")

# ── Log transforms ────────────────────────────────────────────────────────────
cat("\n--- Applying log1p() transforms ---\n")

vars_to_transform <- jb_table$variable[jb_table$reject_H0]

if (length(vars_to_transform) == 0) {
  cat("No variables flagged for transformation (all p >= 0.05)\n")
} else {
  cat(sprintf("Transforming %d variable(s):\n", length(vars_to_transform)))
  for (var in vars_to_transform) {
    new_col <- paste0("log1p_", var)
    df3[[new_col]] <- log1p(df3[[var]])
    cat(sprintf("  log1p(%s) -> %s\n", var, new_col))
  }
}

# ── Save prepped dataset ──────────────────────────────────────────────────────
cat("\n--- Saving prepped dataset ---\n")
saveRDS(df3, file = "analysis/df3_prepped.rds")
cat("Saved: analysis/df3_prepped.rds\n")

# ── Final column check ────────────────────────────────────────────────────────
cat("\n--- str() of prepped dataframe ---\n")
str(df3)

cat("\n=== regression_prep.R complete ===\n")
cat("Next step: open analysis/jb_diagnostics.csv, update CONTROLS in models.R\n")
cat("           to use log1p_ prefix for any row where reject_H0 == TRUE,\n")
cat("           then run models.R.\n")
