# =============================================================================
# sensitivity_calibration.R
# Phase D — Regression calibration for classifier measurement error
#
# Correction factors from final BERT-base Stage 2 test-set confusion matrices
# (Near/Far binary, N/A excluded):
#
#   Social:       sensitivity = 0.896, specificity = 0.579,
#                 FPR = 0.421, Youden_J = 0.475
#   Hypothetical: sensitivity = 0.667, specificity = 0.740,
#                 FPR = 0.260, Youden_J = 0.407
#
# Correction formula: A_corrected = (A_raw - FPR) / Youden_J
# Values outside [0, 1] after correction are clamped.
#
# Prerequisites: run analysis/regression_prep.R and analysis/models.R first.
# =============================================================================

# ── Packages ──────────────────────────────────────────────────────────────────
if (!requireNamespace("survival", quietly = TRUE)) install.packages("survival")
if (!requireNamespace("sandwich", quietly = TRUE)) install.packages("sandwich")
if (!requireNamespace("lmtest",   quietly = TRUE)) install.packages("lmtest")

library(survival)
library(sandwich)
library(lmtest)

# ── Calibration parameters ────────────────────────────────────────────────────
SOC_FPR      <- 0.421
SOC_YOUDEN   <- 0.475   # sensitivity + specificity - 1 = 0.896 + 0.579 - 1

HYP_FPR      <- 0.260
HYP_YOUDEN   <- 0.407   # 0.667 + 0.740 - 1

# ── CONTROLS vector (VIF-pruned 9-variable set) ───────────────────────────────
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

df3$y <- as.integer(df3$is_case)

# ── Apply regression calibration correction ───────────────────────────────────
cat("\n--- Applying calibration correction ---\n")
cat(sprintf("Social:       FPR = %.3f, Youden_J = %.3f\n", SOC_FPR, SOC_YOUDEN))
cat(sprintf("Hypothetical: FPR = %.3f, Youden_J = %.3f\n", HYP_FPR, HYP_YOUDEN))

df3$A_social_cal       <- (df3$A_social       - SOC_FPR) / SOC_YOUDEN
df3$A_hypothetical_cal <- (df3$A_hypothetical - HYP_FPR) / HYP_YOUDEN

# Report and clamp out-of-[0,1] values
n_clamp_soc_lo  <- sum(df3$A_social_cal < 0,       na.rm = TRUE)
n_clamp_soc_hi  <- sum(df3$A_social_cal > 1,       na.rm = TRUE)
n_clamp_hyp_lo  <- sum(df3$A_hypothetical_cal < 0, na.rm = TRUE)
n_clamp_hyp_hi  <- sum(df3$A_hypothetical_cal > 1, na.rm = TRUE)

cat(sprintf("\nA_social_cal clamped below 0: %d rows\n",  n_clamp_soc_lo))
cat(sprintf("A_social_cal clamped above 1: %d rows\n",   n_clamp_soc_hi))
cat(sprintf("A_hypothetical_cal clamped below 0: %d rows\n", n_clamp_hyp_lo))
cat(sprintf("A_hypothetical_cal clamped above 1: %d rows\n", n_clamp_hyp_hi))

df3$A_social_cal       <- pmin(pmax(df3$A_social_cal,       0), 1)
df3$A_hypothetical_cal <- pmin(pmax(df3$A_hypothetical_cal, 0), 1)

# =============================================================================
# MODEL 1 — Conditional logistic regression with calibrated indices
# =============================================================================
cat("\n", strrep("=", 70), "\n", sep = "")
cat("MODEL 1 — Conditional Logistic Regression (calibrated A indices)\n")
cat(strrep("=", 70), "\n", sep = "")

f1_cal <- as.formula(paste(
  "y ~",
  paste(c("A_social_cal", "A_hypothetical_cal", CONTROLS), collapse = " + "),
  "+ strata(stratum_id)"
))
cat("\nFormula:\n"); print(f1_cal); cat("\n")

m1_cal <- clogit(f1_cal, data = df3, method = "efron")

m1_cal_vcov_cl <- sandwich::vcovCL(m1_cal, cluster = ~thread_id, data = df3)
m1_cal_robust  <- lmtest::coeftest(m1_cal, vcov = m1_cal_vcov_cl)

saveRDS(
  list(model = m1_cal, robust_coeftest = m1_cal_robust, vcov_cl = m1_cal_vcov_cl,
       soc_fpr = SOC_FPR, soc_youden = SOC_YOUDEN,
       hyp_fpr = HYP_FPR, hyp_youden = HYP_YOUDEN),
  file = "analysis/sensitivity_calibration_m1.rds"
)
cat("Saved: analysis/sensitivity_calibration_m1.rds\n")

# =============================================================================
# COMPARISON — calibrated vs original M1 coefficients
# =============================================================================
cat("\n", strrep("=", 70), "\n", sep = "")
cat("COMPARISON — Calibrated vs Original M1 coefficients\n")
cat(strrep("=", 70), "\n", sep = "")

# Load original M1 for comparison
if (file.exists("analysis/model1_results.rds")) {
  orig <- readRDS("analysis/model1_results.rds")
  m1_orig_robust <- orig$robust_coeftest

  orig_mat <- matrix(m1_orig_robust, ncol = 4)
  rownames(orig_mat) <- rownames(m1_orig_robust)
  colnames(orig_mat) <- c("Estimate", "SE_robust", "z", "p")
  orig_df <- as.data.frame(orig_mat)
  orig_df$term <- rownames(orig_df)
} else {
  cat("WARNING: analysis/model1_results.rds not found — original comparison skipped\n")
  orig_df <- NULL
}

cal_mat <- matrix(m1_cal_robust, ncol = 4)
rownames(cal_mat) <- rownames(m1_cal_robust)
colnames(cal_mat) <- c("Estimate", "SE_robust", "z", "p")
cal_df <- as.data.frame(cal_mat)
cal_df$term <- rownames(cal_df)

pairs <- list(
  list(orig_term = "A_social",       cal_term = "A_social_cal",       label = "A_social"),
  list(orig_term = "A_hypothetical", cal_term = "A_hypothetical_cal", label = "A_hypothetical")
)

for (pair in pairs) {
  r_cal <- cal_df[cal_df$term == pair$cal_term, ]
  cat(sprintf("\n%s\n", pair$label))
  if (!is.null(orig_df)) {
    r_orig <- orig_df[orig_df$term == pair$orig_term, ]
    cat(sprintf("  Original  (raw A):  Est = %.4f  SE = %.4f  z = %.3f  p = %.4f\n",
                r_orig$Estimate, r_orig$SE_robust, r_orig$z, r_orig$p))
  }
  cat(sprintf("  Calibrated:         Est = %.4f  SE = %.4f  z = %.3f  p = %.4f\n",
              r_cal$Estimate, r_cal$SE_robust, r_cal$z, r_cal$p))
}

cat("\n=== sensitivity_calibration.R complete ===\n")
