# =============================================================================
# model0_baseline.R
# Phase D — Unadjusted conditional logistic regression (baseline for H5)
#
# Exposures only: A_social + A_hypothetical, no surface controls.
# Directly comparable to Model 1 in models.R (which adds 9 controls).
#
# Prerequisites: run analysis/regression_prep.R first to generate df3_prepped.rds
# =============================================================================

# ── Packages ──────────────────────────────────────────────────────────────────
if (!requireNamespace("survival", quietly = TRUE)) install.packages("survival")
if (!requireNamespace("sandwich", quietly = TRUE)) install.packages("sandwich")
if (!requireNamespace("lmtest",   quietly = TRUE)) install.packages("lmtest")

library(survival)
library(sandwich)
library(lmtest)

# ── Load prepped data ─────────────────────────────────────────────────────────
cat("\n--- Loading analysis/df3_prepped.rds ---\n")
df3 <- readRDS("analysis/df3_prepped.rds")
cat(sprintf("Loaded: %d rows, %d columns\n", nrow(df3), ncol(df3)))

if (nrow(df3) != 23096) {
  stop(sprintf("ASSERTION FAILED: expected 23,096 rows, got %d", nrow(df3)))
}
cat("OK  nrow == 23,096\n")

df3$y <- as.integer(df3$is_case)

# =============================================================================
# MODEL 0 — Unadjusted conditional logistic regression
# Exposures only; no surface controls.
# =============================================================================
cat("\n", strrep("=", 70), "\n", sep = "")
cat("MODEL 0 — Unadjusted Conditional Logistic Regression (baseline)\n")
cat(strrep("=", 70), "\n", sep = "")

f0 <- y ~ A_social + A_hypothetical + strata(stratum_id)
cat("\nFormula:\n"); print(f0); cat("\n")

m0 <- clogit(f0, data = df3, method = "efron")

cat("\n--- Standard summary (M0) ---\n")
print(summary(m0))

# ── Cluster-robust SEs at thread_id level ─────────────────────────────────────
cat("\n--- Cluster-robust SEs (clustered at thread_id) ---\n")
vcov_cl   <- sandwich::vcovCL(m0, cluster = ~thread_id, data = df3)
m0_robust <- lmtest::coeftest(m0, vcov = vcov_cl)
print(m0_robust)

# ── Per-variable summary table ────────────────────────────────────────────────
cat("\n--- A_social and A_hypothetical: estimate, SE, z, p, OR, OR 95% CI ---\n\n")

m0_mat <- matrix(m0_robust, ncol = 4)
rownames(m0_mat) <- rownames(m0_robust)
colnames(m0_mat) <- c("Estimate", "SE_robust", "z", "p")
m0_df <- as.data.frame(m0_mat)
m0_df$term <- rownames(m0_df)

for (var in c("A_social", "A_hypothetical")) {
  r <- m0_df[m0_df$term == var, ]
  ci_lo <- r$Estimate - 1.96 * r$SE_robust
  ci_hi <- r$Estimate + 1.96 * r$SE_robust
  cat(sprintf(
    "%s\n  Est = %.4f  SE = %.4f  z = %.3f  p = %.4f\n  OR = %.4f  OR 95%% CI = [%.4f, %.4f]\n\n",
    var,
    r$Estimate, r$SE_robust, r$z, r$p,
    exp(r$Estimate), exp(ci_lo), exp(ci_hi)
  ))
}

# ── Save ──────────────────────────────────────────────────────────────────────
saveRDS(list(model = m0, robust_coeftest = m0_robust, vcov_cl = vcov_cl),
        file = "analysis/model0_results.rds")
cat("Saved: analysis/model0_results.rds\n")

cat("\n=== model0_baseline.R complete ===\n")
