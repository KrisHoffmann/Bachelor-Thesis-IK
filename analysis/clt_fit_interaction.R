# =============================================================================
# clt_fit_interaction.R
# CLT Fit-Account Interaction Test
#
# Tests whether persuasion is maximised when a comment's construal level
# MATCHES the OP's construal level (the "fit" account).
#
# Prerequisites: run analysis/regression_prep.R first to generate df3_prepped.rds
# =============================================================================

set.seed(42)

# ── Packages ──────────────────────────────────────────────────────────────────
for (pkg in c("survival", "sandwich", "lmtest", "dplyr", "arrow")) {
  if (!requireNamespace(pkg, quietly = TRUE)) install.packages(pkg)
  library(pkg, character.only = TRUE)
}

# ── Controls (must match models.R exactly) ────────────────────────────────────
CONTROLS <- c(
  "log1p_noun_to_verb_ratio",
  "log1p_type_token_ratio",
  "log1p_flesch_kincaid",
  "log1p_hedge_density",
  "log1p_mean_sentence_length",
  "log1p_mean_word_length",
  "log1p_punctuation_density",
  "log1p_paragraph_count",
  "log1p_num_urls"
)

# ── Paths ─────────────────────────────────────────────────────────────────────
OP_INDICES_PATH <- "final_dataset_creation/outputs/op_selftexts/op_abstraction_indices.parquet"
OUTPUT_FILE     <- "analysis/output/fit_interaction_results.txt"
OUTPUT_RDS      <- "analysis/fit_interaction_models.rds"

# =============================================================================
# Load and prep matched sample
# =============================================================================
cat("\n", strrep("=", 70), "\n", sep = "")
cat("LOADING DATA\n")
cat(strrep("=", 70), "\n", sep = "")

if (file.exists("analysis/df3_prepped.rds")) {
  cat("Loading analysis/df3_prepped.rds ...\n")
  df3 <- readRDS("analysis/df3_prepped.rds")
  df3$y <- as.integer(df3$is_case)
} else {
  # Replicate regression_prep.R inline (for running without the RDS)
  cat("df3_prepped.rds not found — loading from parquet and applying prep transforms ...\n")
  df3 <- arrow::read_parquet("outputs/stage9_matched_1to3.parquet")
  df3$y <- as.integer(df3$is_case)

  # Impute noun_to_verb_ratio
  if (anyNA(df3$noun_to_verb_ratio)) {
    df3$noun_to_verb_ratio[is.na(df3$noun_to_verb_ratio)] <-
      mean(df3$noun_to_verb_ratio, na.rm = TRUE)
  }

  # Log1p transforms (all 9 controls used in CONTROLS are transformed)
  raw_controls <- c(
    "noun_to_verb_ratio", "type_token_ratio", "flesch_kincaid",
    "hedge_density", "mean_sentence_length", "mean_word_length",
    "punctuation_density", "paragraph_count", "num_urls"
  )
  for (v in raw_controls) df3[[paste0("log1p_", v)]] <- log1p(df3[[v]])
}

cat(sprintf("Matched sample: %d rows, %d cases, %d controls\n",
            nrow(df3), sum(df3$y), sum(df3$y == 0)))

# Verify controls exist
missing_cols <- setdiff(CONTROLS, names(df3))
if (length(missing_cols) > 0) {
  stop(sprintf("Missing control columns: %s", paste(missing_cols, collapse = ", ")))
}

# =============================================================================
# Load OP abstraction indices and join
# =============================================================================
cat("\n", strrep("=", 70), "\n", sep = "")
cat("JOINING OP ABSTRACTION INDICES\n")
cat(strrep("=", 70), "\n", sep = "")

if (!file.exists(OP_INDICES_PATH)) {
  stop(paste("OP indices file not found:", OP_INDICES_PATH))
}

op <- arrow::read_parquet(OP_INDICES_PATH)
cat(sprintf("OP indices loaded: %d threads\n", nrow(op)))
cat(sprintf("  A_Soc_OP defined: %d / %d\n", sum(!is.na(op$A_Soc_OP)), nrow(op)))
cat(sprintf("  A_H_OP   defined: %d / %d\n", sum(!is.na(op$A_H_OP)),   nrow(op)))

# Keep only the columns needed for the join
op_join <- op[, c("thread_id", "A_Soc_OP", "A_H_OP")]

df4 <- dplyr::left_join(df3, op_join, by = "thread_id")

# Join diagnostics at the stratum level
strata_with_op <- df4 |>
  dplyr::group_by(stratum_id) |>
  dplyr::summarise(
    has_op = all(!is.na(A_Soc_OP) & !is.na(A_H_OP)),
    .groups = "drop"
  )

n_strata_total   <- nrow(strata_with_op)
n_strata_with_op <- sum(strata_with_op$has_op)
n_strata_dropped <- n_strata_total - n_strata_with_op

cat(sprintf("\nStrata total          : %d\n", n_strata_total))
cat(sprintf("Strata with OP match  : %d\n", n_strata_with_op))
cat(sprintf("Strata dropped (no OP): %d\n", n_strata_dropped))

# Drop strata without OP data (clogit requires complete contrast within each stratum)
keep_strata <- strata_with_op$stratum_id[strata_with_op$has_op]
df4 <- df4[df4$stratum_id %in% keep_strata, ]

cat(sprintf("Rows after stratum drop: %d (%d cases, %d controls)\n",
            nrow(df4), sum(df4$y), sum(df4$y == 0)))

# Compute mismatch predictors for Model F3
df4$mismatch_soc <- abs(df4$A_social    - df4$A_Soc_OP)
df4$mismatch_hyp <- abs(df4$A_hypothetical - df4$A_H_OP)

# =============================================================================
# Helper: format a coeftest + vcov into thesis-ready tables
# =============================================================================
format_model_table <- function(model, vcov_cl, model_label) {
  ct  <- lmtest::coeftest(model, vcov = vcov_cl)
  est <- ct[, 1]
  se  <- ct[, 2]
  pv  <- ct[, 4]

  or     <- exp(est)
  ci_lo  <- exp(est - 1.96 * se)
  ci_hi  <- exp(est + 1.96 * se)

  fmt_p <- function(p) {
    if      (p < 0.001) "< .001"
    else if (p < 0.01)  sprintf("%.3f", p)
    else                sprintf("%.3f", p)
  }

  header <- sprintf(
    "\n%-34s %8s %8s %8s   %8s %16s\n%s",
    model_label, "log-OR", "SE", "p", "OR", "95% CI",
    strrep("-", 85)
  )
  rows <- mapply(function(nm, b, s, p, o, lo, hi) {
    sprintf("  %-32s %+8.4f %8.4f %8s   %8.4f  [%6.4f, %6.4f]",
            nm, b, s, fmt_p(p), o, lo, hi)
  }, rownames(ct), est, se, pv, or, ci_lo, ci_hi)

  paste(c(header, rows, strrep("-", 85)), collapse = "\n")
}

# =============================================================================
# Open output sink
# =============================================================================
dir.create("analysis/output", showWarnings = FALSE, recursive = TRUE)
sink(OUTPUT_FILE)

cat("CLT FIT-ACCOUNT INTERACTION ANALYSIS\n")
cat(strrep("=", 85), "\n")
cat(sprintf("Date        : %s\n", Sys.time()))
cat(sprintf("Sample      : %d rows (%d cases, %d controls) after OP join\n",
            nrow(df4), sum(df4$y), sum(df4$y == 0)))
cat(sprintf("Strata      : %d (dropped %d with no OP match)\n",
            n_strata_with_op, n_strata_dropped))
cat(sprintf("Cluster SEs : thread_id\n"))
cat(sprintf("Method      : clogit (Efron), strata(stratum_id)\n"))
cat(strrep("=", 85), "\n\n")

# =============================================================================
# Model F1 — OP main effects
# =============================================================================
cat(strrep("=", 85), "\n")
cat("MODEL F1 — OP Main Effects\n")
cat("Tests whether OP construal level independently predicts delta award\n")
cat("above and beyond comment-level construal.\n")
cat(strrep("=", 85), "\n")

f1_terms <- c("A_social", "A_hypothetical", "A_Soc_OP", "A_H_OP", CONTROLS)
fF1 <- as.formula(paste("y ~", paste(f1_terms, collapse = " + "), "+ strata(stratum_id)"))
cat("\nFormula:\n"); print(fF1)

mF1        <- clogit(fF1, data = df4, method = "efron")
vcov_F1    <- sandwich::vcovCL(mF1, cluster = ~thread_id, data = df4)

cat(format_model_table(mF1, vcov_F1, "MODEL F1"))

cat("\n\nINTERPRETATION (F1):\n")
ctF1 <- lmtest::coeftest(mF1, vcov = vcov_F1)
b_soc_op <- ctF1["A_Soc_OP", 1]; p_soc_op <- ctF1["A_Soc_OP", 4]
b_hyp_op <- ctF1["A_H_OP",   1]; p_hyp_op <- ctF1["A_H_OP",   4]
cat(sprintf(
  "  A_Soc_OP: log-OR = %+.4f, %s\n    -> OP social abstraction %s associated with delta (%s).\n",
  b_soc_op,
  ifelse(p_soc_op < .05, sprintf("p = %.3f *", p_soc_op), sprintf("p = %.3f (n.s.)", p_soc_op)),
  ifelse(b_soc_op > 0, "positively", "negatively"),
  ifelse(p_soc_op < .05, "significant", "not significant")
))
cat(sprintf(
  "  A_H_OP  : log-OR = %+.4f, %s\n    -> OP hypothetical abstraction %s associated with delta (%s).\n",
  b_hyp_op,
  ifelse(p_hyp_op < .05, sprintf("p = %.3f *", p_hyp_op), sprintf("p = %.3f (n.s.)", p_hyp_op)),
  ifelse(b_hyp_op > 0, "positively", "negatively"),
  ifelse(p_hyp_op < .05, "significant", "not significant")
))
if (p_soc_op >= .05 && p_hyp_op >= .05) {
  cat("  Neither OP index is significant: OP construal level does not independently\n")
  cat("  predict persuasion. This is consistent with a pure comment-level effect\n")
  cat("  rather than OP susceptibility driving delta awards.\n")
} else {
  cat("  At least one OP index is significant: OP construal level influences\n")
  cat("  persuasion independently of comment construal, suggesting OP susceptibility.\n")
}

# =============================================================================
# Model F2 — Interaction / match
# =============================================================================
cat("\n\n", strrep("=", 85), "\n", sep = "")
cat("MODEL F2 — Construal Match Interaction\n")
cat("Fit account predicts POSITIVE interaction coefficients:\n")
cat("persuasion is highest when comment and OP construal levels are aligned.\n")
cat(strrep("=", 85), "\n")

f2_terms <- c("A_social", "A_hypothetical", "A_Soc_OP", "A_H_OP",
              "A_social:A_Soc_OP", "A_hypothetical:A_H_OP", CONTROLS)
fF2 <- as.formula(paste("y ~", paste(f2_terms, collapse = " + "), "+ strata(stratum_id)"))
cat("\nFormula:\n"); print(fF2)

mF2        <- clogit(fF2, data = df4, method = "efron")
vcov_F2    <- sandwich::vcovCL(mF2, cluster = ~thread_id, data = df4)

cat(format_model_table(mF2, vcov_F2, "MODEL F2"))

cat("\n\nINTERPRETATION (F2):\n")
ctF2 <- lmtest::coeftest(mF2, vcov = vcov_F2)

# Retrieve interaction terms — handle colon vs multiplication naming
int_soc_nm <- grep("A_social.*A_Soc_OP|A_Soc_OP.*A_social", rownames(ctF2), value = TRUE)[1]
int_hyp_nm <- grep("A_hypothetical.*A_H_OP|A_H_OP.*A_hypothetical", rownames(ctF2), value = TRUE)[1]

for (nm in c(int_soc_nm, int_hyp_nm)) {
  if (!is.na(nm) && nm %in% rownames(ctF2)) {
    b <- ctF2[nm, 1]; p <- ctF2[nm, 4]
    dim_label <- if (grepl("Soc", nm)) "Social" else "Hypothetical"
    cat(sprintf(
      "  %s interaction (%s): log-OR = %+.4f, %s\n    -> %s\n",
      dim_label, nm, b,
      ifelse(p < .05, sprintf("p = %.3f *", p), sprintf("p = %.3f (n.s.)", p)),
      ifelse(p < .05,
        ifelse(b > 0,
          "SUPPORTS fit account: matched construal boosts persuasion.",
          "CONTRADICTS fit account: matched construal reduces persuasion."),
        "No evidence for construal match effect on this dimension.")
    ))
  }
}

# =============================================================================
# Model F3 — Continuous mismatch distance
# =============================================================================
cat("\n\n", strrep("=", 85), "\n", sep = "")
cat("MODEL F3 — Continuous Mismatch Distance\n")
cat("Fit account predicts NEGATIVE distance coefficients:\n")
cat("smaller |comment - OP| distance -> better match -> higher delta odds.\n")
cat(strrep("=", 85), "\n")

f3_terms <- c("A_social", "A_hypothetical", "mismatch_soc", "mismatch_hyp", CONTROLS)
fF3 <- as.formula(paste("y ~", paste(f3_terms, collapse = " + "), "+ strata(stratum_id)"))
cat("\nFormula:\n"); print(fF3)

mF3        <- clogit(fF3, data = df4, method = "efron")
vcov_F3    <- sandwich::vcovCL(mF3, cluster = ~thread_id, data = df4)

cat(format_model_table(mF3, vcov_F3, "MODEL F3"))

cat("\n\nINTERPRETATION (F3):\n")
ctF3 <- lmtest::coeftest(mF3, vcov = vcov_F3)
for (nm in c("mismatch_soc", "mismatch_hyp")) {
  if (nm %in% rownames(ctF3)) {
    b <- ctF3[nm, 1]; p <- ctF3[nm, 4]
    dim_label <- if (nm == "mismatch_soc") "Social" else "Hypothetical"
    cat(sprintf(
      "  %s mismatch (%s): log-OR = %+.4f, %s\n    -> %s\n",
      dim_label, nm, b,
      ifelse(p < .05, sprintf("p = %.3f *", p), sprintf("p = %.3f (n.s.)", p)),
      ifelse(p < .05,
        ifelse(b < 0,
          "SUPPORTS fit account: smaller construal distance -> higher delta odds.",
          "CONTRADICTS fit account: smaller distance associated with lower delta odds."),
        "No evidence for construal distance effect on this dimension.")
    ))
  }
}

# =============================================================================
# Overall synthesis
# =============================================================================
cat("\n\n", strrep("=", 85), "\n", sep = "")
cat("OVERALL SYNTHESIS\n")
cat(strrep("=", 85), "\n")
cat(
"Three complementary tests of the construal fit account:\n\n",
"F1 (OP main effects)  — tests whether OP framing predicts susceptibility\n",
"                         independently of comment construal.\n",
"F2 (match interaction) — tests the core fit prediction: comment*OP alignment\n",
"                         should produce a positive interaction coefficient.\n",
"F3 (mismatch distance) — tests fit via |comment - OP| distance; negative\n",
"                         coefficient = closer match = higher persuasion odds.\n\n",
"If F1 is null but F2/F3 are significant: pure fit effect (no susceptibility main effect).\n",
"If F1 is significant but F2/F3 are null: OP construal acts as a main-effect moderator,\n",
"  not a match target — inconsistent with construal fit, consistent with a ceiling/floor\n",
"  interpretation (e.g., highly abstract OPs are simply harder to persuade).\n",
"If all three are null: comment-level construal (from M1) is the full story;\n",
"  OP construal does not additionally account for variation in delta awards.\n",
sep = "")

sink()

cat(sprintf("\nResults saved: %s\n", OUTPUT_FILE))

# =============================================================================
# Save model objects
# =============================================================================
saveRDS(
  list(
    data   = df4,
    mF1    = mF1, vcov_F1 = vcov_F1,
    mF2    = mF2, vcov_F2 = vcov_F2,
    mF3    = mF3, vcov_F3 = vcov_F3
  ),
  file = OUTPUT_RDS
)
cat(sprintf("Models saved : %s\n", OUTPUT_RDS))

# Also print to console so RStudio shows results immediately
cat("\n"); cat(readLines(OUTPUT_FILE), sep = "\n")

cat("\n=== clt_fit_interaction.R complete ===\n")
