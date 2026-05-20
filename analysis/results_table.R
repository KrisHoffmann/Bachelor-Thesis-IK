# =============================================================================
# results_table.R
# Phase D — Core thesis table from saved model objects
#
# Prerequisites: run analysis/models.R first to generate:
#   analysis/model1_results.rds
#   analysis/model3_results.rds
# =============================================================================

# ── Load saved model objects ──────────────────────────────────────────────────
m1_saved <- readRDS("analysis/model1_results.rds")
m3_saved <- readRDS("analysis/model3_results.rds")

m1_robust <- m1_saved$robust_coeftest
m1        <- m1_saved$model
m3        <- m3_saved$model

cat(sprintf("M3 author random effect dropped: %s\n", m3_saved$author_dropped))
if (!is.na(m3_saved$author_var)) {
  cat(sprintf("M3 author variance: %.6f\n", m3_saved$author_var))
}

# =============================================================================
# CORE THESIS TABLE — A_social and A_hypothetical across M1 and M3
# =============================================================================
cat("\n", strrep("=", 70), "\n", sep = "")
cat("CORE THESIS TABLE — A_social and A_hypothetical coefficients\n")
cat(strrep("=", 70), "\n", sep = "")

target_vars <- c("A_social", "A_hypothetical")

# M1 robust estimates
# as.data.frame() on a coeftest object collapses to one column; use matrix() instead
m1_mat <- matrix(m1_robust, ncol = 4)
rownames(m1_mat) <- rownames(m1_robust)
colnames(m1_mat) <- c("est_m1", "se_m1_robust", "z_m1", "p_m1")
m1_rob_tbl <- as.data.frame(m1_mat)
m1_rob_tbl$term <- rownames(m1_rob_tbl)

# M3 estimates
m3_coefs <- as.data.frame(summary(m3)$coefficients)
colnames(m3_coefs) <- c("est_m3", "se_m3", "z_m3", "p_m3")
m3_coefs$term <- rownames(m3_coefs)

build_row <- function(var) {
  r1 <- m1_rob_tbl[m1_rob_tbl$term == var, ]
  r3 <- m3_coefs[m3_coefs$term == var, ]

  # M1 CIs (based on robust SE)
  ci95_lo_m1 <- r1$est_m1 - 1.96  * r1$se_m1_robust
  ci95_hi_m1 <- r1$est_m1 + 1.96  * r1$se_m1_robust
  ci99_lo_m1 <- r1$est_m1 - 2.576 * r1$se_m1_robust
  ci99_hi_m1 <- r1$est_m1 + 2.576 * r1$se_m1_robust

  data.frame(
    Variable        = var,
    # M1
    M1_Est          = round(r1$est_m1,       4),
    M1_SE_robust    = round(r1$se_m1_robust, 4),
    M1_z            = round(r1$z_m1,         3),
    M1_p            = signif(r1$p_m1,        3),
    M1_OR           = round(exp(r1$est_m1),  4),
    M1_CI95_lo      = round(ci95_lo_m1,      4),
    M1_CI95_hi      = round(ci95_hi_m1,      4),
    M1_CI99_lo      = round(ci99_lo_m1,      4),
    M1_CI99_hi      = round(ci99_hi_m1,      4),
    M1_OR_CI95_lo   = round(exp(ci95_lo_m1), 4),
    M1_OR_CI95_hi   = round(exp(ci95_hi_m1), 4),
    # M3
    M3_Est          = round(r3$est_m3,       4),
    M3_SE           = round(r3$se_m3,        4),
    M3_z            = round(r3$z_m3,         3),
    M3_p            = signif(r3$p_m3,        3),
    M3_OR           = round(exp(r3$est_m3),  4),
    stringsAsFactors = FALSE
  )
}

thesis_table <- do.call(rbind, lapply(target_vars, build_row))
rownames(thesis_table) <- NULL

cat("\nCoefficients (log-odds scale):\n\n")
print(thesis_table[, c("Variable",
                        "M1_Est", "M1_SE_robust", "M1_z", "M1_p",
                        "M1_CI95_lo", "M1_CI95_hi",
                        "M1_CI99_lo", "M1_CI99_hi",
                        "M3_Est", "M3_SE", "M3_z", "M3_p")],
      row.names = FALSE)

cat("\nOdds ratios:\n\n")
print(thesis_table[, c("Variable",
                        "M1_OR", "M1_OR_CI95_lo", "M1_OR_CI95_hi",
                        "M3_OR")],
      row.names = FALSE)

# Save thesis table
write.csv(thesis_table, file = "analysis/thesis_table_main.csv", row.names = FALSE)
cat("\nSaved: analysis/thesis_table_main.csv\n")

cat("\n=== results_table.R complete ===\n")
