# =============================================================================
# collinearity_check.R
# Phase D — VIF diagnostics to determine the final control set
#
# Run order: regression_prep.R → collinearity_check.R → models.R
#
# Reads:  analysis/df3_prepped.rds  (output of regression_prep.R)
# Writes: analysis/vif_diagnostics.csv  (full VIF table after first pass)
# =============================================================================

# ── Packages ──────────────────────────────────────────────────────────────────
if (!requireNamespace("car",   quietly = TRUE)) install.packages("car")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")

library(car)
library(dplyr)

# ── Load prepped data ─────────────────────────────────────────────────────────
cat("\n--- Loading analysis/df3_prepped.rds ---\n")
df3 <- readRDS("analysis/df3_prepped.rds")
cat(sprintf("Loaded: %d rows, %d columns\n", nrow(df3), ncol(df3)))

df3$y <- as.integer(df3$is_case)

# ── Starting control set (all 12 log1p controls from models.R) ────────────────
# These mirror the CONTROLS vector in models.R — log1p_ prefix on all 12.
# Adjust if regression_prep.R did not flag some variables for transformation.
CONTROLS_INIT <- c(
  "log1p_noun_to_verb_ratio",
  "log1p_type_token_ratio",
  "log1p_flesch_kincaid",
  "log1p_hedge_density",
  "log1p_mean_sentence_length",
  "log1p_mean_word_length",
  "log1p_punctuation_density",
  "log1p_paragraph_count",
  "log1p_num_sentences",
  "log1p_num_tokens",
  "log1p_num_word_tokens",
  "log1p_num_urls"
)

# Verify columns exist
missing_cols <- setdiff(CONTROLS_INIT, names(df3))
if (length(missing_cols) > 0) {
  stop(sprintf(
    "Missing columns in df3_prepped.rds:\n  %s\n\n%s",
    paste(missing_cols, collapse = ", "),
    "Ensure regression_prep.R ran successfully and all controls were log1p-transformed."
  ))
}

# ── Helper: fit glm and compute VIF ──────────────────────────────────────────
compute_vif <- function(controls, data) {
  f <- as.formula(paste(
    "y ~", paste(c("A_social", "A_hypothetical", controls), collapse = " + ")
  ))
  # Suppress glm convergence noise — we only need VIF, not inference
  fit <- suppressWarnings(glm(f, data = data, family = binomial))
  vif_vals <- car::vif(fit)
  # car::vif returns a named numeric vector for main effects
  data.frame(
    variable = names(vif_vals),
    VIF      = round(as.numeric(vif_vals), 3),
    stringsAsFactors = FALSE
  ) |> arrange(desc(VIF))
}

# ── Pass 1: full VIF table ────────────────────────────────────────────────────
cat("\n", strrep("=", 70), "\n", sep = "")
cat("PASS 1 — VIF on all 12 controls\n")
cat(strrep("=", 70), "\n")

vif_pass1 <- compute_vif(CONTROLS_INIT, df3)
print(vif_pass1, row.names = FALSE)

# Flag thresholds
cat("\nVariables with VIF > 10 (drop recommended):\n")
flag10 <- vif_pass1$variable[vif_pass1$VIF > 10]
if (length(flag10) == 0) cat("  none\n") else cat(paste0("  ", flag10, collapse = "\n"), "\n")

cat("\nVariables with 5 < VIF <= 10 (warn):\n")
flag5 <- vif_pass1$variable[vif_pass1$VIF > 5 & vif_pass1$VIF <= 10]
if (length(flag5) == 0) cat("  none\n") else cat(paste0("  ", flag5, collapse = "\n"), "\n")

# Save initial VIF table
write.csv(vif_pass1, file = "analysis/vif_diagnostics.csv", row.names = FALSE)
cat("\nSaved: analysis/vif_diagnostics.csv\n")

# ── Iterative backward drop: remove highest VIF > 10, rerun until clean ───────
cat("\n", strrep("=", 70), "\n", sep = "")
cat("ITERATIVE VIF REDUCTION (drop highest VIF > 10 per iteration)\n")
cat(strrep("=", 70), "\n")

# Exclude A_social and A_hypothetical from dropping — they are the exposures
CONTROLS_CURRENT <- CONTROLS_INIT
dropped_vars <- character(0)
pass_num <- 1L

repeat {
  vif_tbl <- compute_vif(CONTROLS_CURRENT, df3)

  # Only consider controls for dropping (not the two exposure variables)
  control_vif <- vif_tbl[vif_tbl$variable %in% CONTROLS_CURRENT, ]
  max_vif_row <- control_vif[which.max(control_vif$VIF), ]

  if (max_vif_row$VIF <= 10) {
    cat(sprintf("\nAll remaining controls have VIF <= 10. Stopping at pass %d.\n", pass_num))
    break
  }

  drop_var <- max_vif_row$variable
  cat(sprintf(
    "\nPass %d: highest VIF = %.3f (%s) > 10 — dropping\n",
    pass_num, max_vif_row$VIF, drop_var
  ))
  dropped_vars   <- c(dropped_vars, drop_var)
  CONTROLS_CURRENT <- setdiff(CONTROLS_CURRENT, drop_var)

  cat(sprintf("Controls remaining (%d): %s\n",
              length(CONTROLS_CURRENT),
              paste(CONTROLS_CURRENT, collapse = ", ")))

  cat(sprintf("\nVIF after pass %d:\n", pass_num))
  print(vif_tbl[vif_tbl$variable %in% c("A_social", "A_hypothetical", CONTROLS_CURRENT), ],
        row.names = FALSE)

  pass_num <- pass_num + 1L
}

# ── Final VIF table ───────────────────────────────────────────────────────────
cat("\n", strrep("=", 70), "\n", sep = "")
cat("FINAL VIF TABLE (all variables <= 10)\n")
cat(strrep("=", 70), "\n")

vif_final <- compute_vif(CONTROLS_CURRENT, df3)
print(vif_final, row.names = FALSE)

# Warn on any remaining 5 < VIF <= 10
warn5 <- vif_final$variable[vif_final$VIF > 5 & vif_final$variable %in% CONTROLS_CURRENT]
if (length(warn5) > 0) {
  cat(sprintf(
    "\nWARNING: %d control(s) still have VIF > 5 (moderate collinearity — review):\n  %s\n",
    length(warn5), paste(warn5, collapse = ", ")
  ))
}

# ── Summary of what was dropped ───────────────────────────────────────────────
cat("\n", strrep("=", 70), "\n", sep = "")
cat("SUMMARY\n")
cat(strrep("=", 70), "\n")

if (length(dropped_vars) == 0) {
  cat("No variables were dropped. All 12 controls have VIF <= 10.\n")
} else {
  cat(sprintf("Dropped %d variable(s) due to VIF > 10:\n", length(dropped_vars)))
  cat(paste0("  ", dropped_vars, collapse = "\n"), "\n")
}

cat(sprintf("\nFinal control count: %d\n", length(CONTROLS_CURRENT)))

# ── Print final CONTROLS vector ready to paste into models.R ─────────────────
cat("\n", strrep("=", 70), "\n", sep = "")
cat("COPY THIS INTO models.R — replace the CONTROLS vector\n")
cat(strrep("=", 70), "\n")
cat("\nCONTROLS <- c(\n")
for (i in seq_along(CONTROLS_CURRENT)) {
  comma <- if (i < length(CONTROLS_CURRENT)) "," else ""
  cat(sprintf('  "%s"%s\n', CONTROLS_CURRENT[i], comma))
}
cat(")\n")

cat("\n=== collinearity_check.R complete ===\n")
cat("Next step: paste the CONTROLS vector above into models.R, then run models.R.\n")
