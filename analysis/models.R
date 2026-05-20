# =============================================================================
# models.R
# Phase D — Conditional logistic (M1) and GLMM robustness check (M3)
#
# Prerequisites: run analysis/regression_prep.R first to generate df3_prepped.rds
# =============================================================================

# ── Packages ──────────────────────────────────────────────────────────────────
if (!requireNamespace("survival",  quietly = TRUE)) install.packages("survival")
if (!requireNamespace("lme4",      quietly = TRUE)) install.packages("lme4")
if (!requireNamespace("sandwich",  quietly = TRUE)) install.packages("sandwich")
if (!requireNamespace("lmtest",    quietly = TRUE)) install.packages("lmtest")
if (!requireNamespace("dplyr",     quietly = TRUE)) install.packages("dplyr")

library(survival)
library(lme4)
library(sandwich)
library(lmtest)
library(dplyr)

# =============================================================================
# CONTROLS VECTOR
# UPDATE THIS after running regression_prep.R.
# For any variable where reject_H0 == TRUE in analysis/jb_diagnostics.csv,
# replace its name here with the log1p_ prefixed version.
# Example: if flesch_kincaid is flagged, change "flesch_kincaid" to
#          "log1p_flesch_kincaid".
# =============================================================================
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

# ── Load prepped data ─────────────────────────────────────────────────────────
cat("\n--- Loading analysis/df3_prepped.rds ---\n")
df3 <- readRDS("analysis/df3_prepped.rds")
cat(sprintf("Loaded: %d rows, %d columns\n", nrow(df3), ncol(df3)))

# Outcome must be numeric 0/1 for clogit and glmer
df3$y <- as.integer(df3$is_case)

# Verify all CONTROLS columns exist in df3
missing_cols <- setdiff(CONTROLS, names(df3))
if (length(missing_cols) > 0) {
  stop(sprintf(
    "CONTROLS references columns not found in df3_prepped.rds:\n  %s\n\n%s",
    paste(missing_cols, collapse = ", "),
    "Did you update CONTROLS with log1p_ prefixes after running regression_prep.R?"
  ))
}

cat("\nControls in use:\n")
cat(paste0("  ", CONTROLS, collapse = "\n"), "\n")

# =============================================================================
# MODEL 1 — Conditional logistic regression (primary model)
#
# Grouping: strata(stratum_id)  — one stratum = one case + its matched controls
# Cluster-robust SEs: clustered at thread_id (a subreddit thread can span strata)
#
# NOTE: stratum_id != thread_id. Do NOT use strata(thread_id).
#       See analysis/README.md for the full explanation.
# =============================================================================
cat("\n", strrep("=", 70), "\n", sep = "")
cat("MODEL 1 — Conditional Logistic Regression\n")
cat(strrep("=", 70), "\n", sep = "")

f1 <- as.formula(paste(
  "y ~",
  paste(c("A_social", "A_hypothetical", CONTROLS), collapse = " + "),
  "+ strata(stratum_id)"
))
cat("\nFormula:\n"); print(f1); cat("\n")

m1 <- clogit(f1, data = df3, method = "efron")

cat("\n--- Standard summary (M1) ---\n")
print(summary(m1))

# Cluster-robust SEs at thread_id level
cat("\n--- Cluster-robust SEs (clustered at thread_id) ---\n")
m1_vcov_cl  <- sandwich::vcovCL(m1, cluster = ~thread_id, data = df3)
m1_robust   <- lmtest::coeftest(m1, vcov = m1_vcov_cl)
print(m1_robust)

# Save both objects
saveRDS(list(model = m1, robust_coeftest = m1_robust, vcov_cl = m1_vcov_cl),
        file = "analysis/model1_results.rds")
cat("\nSaved: analysis/model1_results.rds\n")

# =============================================================================
# MODEL 2 — PLACEHOLDER
# Identity not yet confirmed (hierarchical clogit vs mediation analysis).
# Do not implement until supervisor sign-off. See Phase D handoff §3.
# =============================================================================
cat("\n", strrep("=", 70), "\n", sep = "")
cat("MODEL 2 — PLACEHOLDER (not implemented)\n")
cat(strrep("=", 70), "\n", sep = "")
cat("# MODEL 2 is reserved. Awaiting supervisor sign-off on model type.\n")
cat("# Candidate: hierarchical clogit or mediation model.\n")
cat("# See Phase D handoff §3 for decision criteria.\n")

# =============================================================================
# MODEL 3 — GLMM robustness check
#
# Random effects: (1|thread_id) for subreddit thread,
#                 (1|author) for repeated-commenter variance.
# If author variance < 0.01 or convergence fails, refit with thread_id only.
# =============================================================================
cat("\n", strrep("=", 70), "\n", sep = "")
cat("MODEL 3 — GLMM Robustness Check (binomial mixed model)\n")
cat(strrep("=", 70), "\n", sep = "")

f3_full <- as.formula(paste(
  "y ~",
  paste(c("A_social", "A_hypothetical", CONTROLS), collapse = " + "),
  "+ (1|thread_id) + (1|author)"
))
cat("\nFormula (full):\n"); print(f3_full); cat("\n")

m3_converged <- TRUE
m3_author_dropped <- FALSE

m3 <- tryCatch({
  withCallingHandlers(
    glmer(f3_full, data = df3, family = binomial,
          control = glmerControl(optimizer = "bobyqa")),
    warning = function(w) {
      if (grepl("convergence|failed to converge|Model failed", conditionMessage(w),
                ignore.case = TRUE)) {
        cat(sprintf("CONVERGENCE WARNING: %s\n", conditionMessage(w)))
        m3_converged <<- FALSE
      }
      invokeRestart("muffleWarning")
    }
  )
}, error = function(e) {
  cat(sprintf("ERROR fitting full M3: %s\n", conditionMessage(e)))
  m3_converged <<- FALSE
  NULL
})

# Check author random-effect variance
author_var <- NA_real_
if (!is.null(m3)) {
  vc <- as.data.frame(VarCorr(m3))
  author_row <- vc[vc$grp == "author", ]
  if (nrow(author_row) > 0) {
    author_var <- author_row$vcov[1]
    cat(sprintf("\nauthor random-effect variance: %.6f\n", author_var))
  }
}

refit_without_author <- (!m3_converged) || (!is.na(author_var) && author_var < 0.01)

if (refit_without_author) {
  if (!m3_converged) cat("Refitting without (1|author): convergence issue in full model\n")
  if (!is.na(author_var) && author_var < 0.01)
    cat(sprintf("Refitting without (1|author): variance = %.6f < 0.01 threshold\n", author_var))

  m3_author_dropped <- TRUE
  f3_reduced <- as.formula(paste(
    "y ~",
    paste(c("A_social", "A_hypothetical", CONTROLS), collapse = " + "),
    "+ (1|thread_id)"
  ))
  cat("\nFormula (reduced, thread_id only):\n"); print(f3_reduced); cat("\n")

  m3 <- glmer(f3_reduced, data = df3, family = binomial,
              control = glmerControl(optimizer = "bobyqa"))
}

cat("\n--- Summary (M3) ---\n")
print(summary(m3))

saveRDS(list(model = m3, author_dropped = m3_author_dropped, author_var = author_var),
        file = "analysis/model3_results.rds")
cat("\nSaved: analysis/model3_results.rds\n")

# =============================================================================
# CORE THESIS TABLE — A_social and A_hypothetical across M1 and M3
# =============================================================================
cat("\n", strrep("=", 70), "\n", sep = "")
cat("CORE THESIS TABLE — A_social and A_hypothetical coefficients\n")
cat(strrep("=", 70), "\n", sep = "")

target_vars <- c("A_social", "A_hypothetical")

# M1 robust estimates
m1_rob_tbl <- as.data.frame(m1_robust)
m1_rob_tbl$term <- rownames(m1_rob_tbl)
colnames(m1_rob_tbl) <- c("est_m1", "se_m1_robust", "z_m1", "p_m1", "term")

# M3 estimates
m3_coefs <- as.data.frame(summary(m3)$coefficients)
m3_coefs$term <- rownames(m3_coefs)
colnames(m3_coefs) <- c("est_m3", "se_m3", "z_m3", "p_m3", "term")

build_row <- function(var) {
  r1 <- m1_rob_tbl[m1_rob_tbl$term == var, ]
  r3 <- m3_coefs[m3_coefs$term == var, ]

  # M1 CIs (based on robust SE)
  ci95_lo_m1 <- r1$est_m1 - 1.96 * r1$se_m1_robust
  ci95_hi_m1 <- r1$est_m1 + 1.96 * r1$se_m1_robust
  ci99_lo_m1 <- r1$est_m1 - 2.576 * r1$se_m1_robust
  ci99_hi_m1 <- r1$est_m1 + 2.576 * r1$se_m1_robust

  data.frame(
    Variable      = var,
    # M1
    M1_Est        = round(r1$est_m1,        4),
    M1_SE_robust  = round(r1$se_m1_robust,  4),
    M1_z          = round(r1$z_m1,          3),
    M1_p          = signif(r1$p_m1,         3),
    M1_OR         = round(exp(r1$est_m1),   4),
    M1_CI95       = sprintf("[%.3f, %.3f]", ci95_lo_m1, ci95_hi_m1),
    M1_CI99       = sprintf("[%.3f, %.3f]", ci99_lo_m1, ci99_hi_m1),
    M1_OR_CI95    = sprintf("[%.3f, %.3f]", exp(ci95_lo_m1), exp(ci95_hi_m1)),
    # M3
    M3_Est        = round(r3$est_m3,        4),
    M3_SE         = round(r3$se_m3,         4),
    M3_z          = round(r3$z_m3,          3),
    M3_p          = signif(r3$p_m3,         3),
    M3_OR         = round(exp(r3$est_m3),   4),
    stringsAsFactors = FALSE
  )
}

thesis_table <- do.call(rbind, lapply(target_vars, build_row))
rownames(thesis_table) <- NULL

cat("\nCoefficients (log-odds scale):\n\n")
print(thesis_table[, c("Variable",
                        "M1_Est", "M1_SE_robust", "M1_z", "M1_p",
                        "M1_CI95", "M1_CI99",
                        "M3_Est", "M3_SE", "M3_z", "M3_p")],
      row.names = FALSE)

cat("\nOdds ratios:\n\n")
print(thesis_table[, c("Variable",
                        "M1_OR", "M1_OR_CI95",
                        "M3_OR")],
      row.names = FALSE)

# Save thesis table
write.csv(thesis_table, file = "analysis/thesis_table_main.csv", row.names = FALSE)
cat("\nSaved: analysis/thesis_table_main.csv\n")

cat("\n=== models.R complete ===\n")
