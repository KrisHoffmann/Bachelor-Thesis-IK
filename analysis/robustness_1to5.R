# =============================================================================
# robustness_1to5.R
# Phase D — Robustness check: 1:5 matched sample
#
# Prerequisites: stage9_matched_1to5.parquet must exist in DATA_DIR
# =============================================================================

# ── Packages ──────────────────────────────────────────────────────────────────
if (!requireNamespace("arrow",    quietly = TRUE)) install.packages("arrow")
if (!requireNamespace("survival", quietly = TRUE)) install.packages("survival")
if (!requireNamespace("lme4",     quietly = TRUE)) install.packages("lme4")
if (!requireNamespace("sandwich", quietly = TRUE)) install.packages("sandwich")
if (!requireNamespace("lmtest",   quietly = TRUE)) install.packages("lmtest")
if (!requireNamespace("dplyr",    quietly = TRUE)) install.packages("dplyr")

library(arrow)
library(survival)
library(lme4)
library(sandwich)
library(lmtest)
library(dplyr)

DATA_DIR <- "C:/Users/kris/Documents/Thesis_R"

# ── Load data ─────────────────────────────────────────────────────────────────
cat("\n--- Loading stage9_matched_1to5.parquet ---\n")
df5 <- arrow::read_parquet(file.path(DATA_DIR, "stage9_matched_1to5.parquet"))
cat(sprintf("Loaded: %d rows, %d columns\n", nrow(df5), ncol(df5)))

# ── Assertions ────────────────────────────────────────────────────────────────
if (nrow(df5) != 33890) {
  stop(sprintf("ASSERTION FAILED: expected 33,890 rows, got %d", nrow(df5)))
}
cat("OK  nrow == 33,890\n")

n_cases <- sum(df5$is_case, na.rm = TRUE)
if (n_cases != 5849) {
  stop(sprintf("ASSERTION FAILED: expected 5,849 cases, got %d", n_cases))
}
cat("OK  sum(is_case) == 5,849\n")

# ── Impute noun_to_verb_ratio ─────────────────────────────────────────────────
cat("\n--- Imputing noun_to_verb_ratio ---\n")
n_null_ntvr <- sum(is.na(df5$noun_to_verb_ratio))
cat(sprintf("  NAs in matched sample: %d\n", n_null_ntvr))

if (n_null_ntvr > 0) {
  corpus_mean_ntvr <- mean(df5$noun_to_verb_ratio, na.rm = TRUE)
  cat(sprintf("  Imputing with matched-sample mean: %.4f\n", corpus_mean_ntvr))
  df5$noun_to_verb_ratio <- ifelse(
    is.na(df5$noun_to_verb_ratio),
    corpus_mean_ntvr,
    df5$noun_to_verb_ratio
  )
} else {
  cat("  No NAs found — no imputation needed\n")
}

# ── Apply log1p transforms ────────────────────────────────────────────────────
cat("\n--- Applying log1p() transforms ---\n")
log1p_vars <- c(
  "noun_to_verb_ratio", "type_token_ratio", "flesch_kincaid",
  "hedge_density", "mean_sentence_length", "mean_word_length",
  "punctuation_density", "paragraph_count", "num_sentences",
  "num_tokens", "num_word_tokens", "num_urls"
)
for (var in log1p_vars) {
  df5[[paste0("log1p_", var)]] <- log1p(df5[[var]])
}
cat(sprintf("  Created log1p_ columns for %d variables\n", length(log1p_vars)))

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

missing_cols <- setdiff(c(CONTROLS, "A_social", "A_hypothetical",
                           "stratum_id", "thread_id", "author", "is_case"),
                        names(df5))
if (length(missing_cols) > 0) {
  stop(sprintf("Missing columns: %s", paste(missing_cols, collapse = ", ")))
}

df5$y <- as.integer(df5$is_case)

# =============================================================================
# MODEL 1 — Conditional logistic regression
# =============================================================================
cat("\n", strrep("=", 70), "\n", sep = "")
cat("MODEL 1 — Conditional Logistic Regression (1:5 sample)\n")
cat(strrep("=", 70), "\n", sep = "")

f1 <- as.formula(paste(
  "y ~",
  paste(c("A_social", "A_hypothetical", CONTROLS), collapse = " + "),
  "+ strata(stratum_id)"
))
cat("\nFormula:\n"); print(f1); cat("\n")

m1 <- clogit(f1, data = df5, method = "efron")

m1_vcov_cl <- sandwich::vcovCL(m1, cluster = ~thread_id, data = df5)
m1_robust  <- lmtest::coeftest(m1, vcov = m1_vcov_cl)

saveRDS(list(model = m1, robust_coeftest = m1_robust, vcov_cl = m1_vcov_cl),
        file = "analysis/robustness_1to5_m1.rds")
cat("Saved: analysis/robustness_1to5_m1.rds\n")

# =============================================================================
# MODEL 3 — GLMM robustness check
# =============================================================================
cat("\n", strrep("=", 70), "\n", sep = "")
cat("MODEL 3 — GLMM Robustness Check (1:5 sample)\n")
cat(strrep("=", 70), "\n", sep = "")

n_unique_authors <- length(unique(df5$author))
cat(sprintf("\nUnique authors in df5: %d\n", n_unique_authors))
if (n_unique_authors > 5000) {
  # Author random effect has many levels (>5,000); convergence may be slow.
}

f3_full <- as.formula(paste(
  "y ~",
  paste(c("A_social", "A_hypothetical", CONTROLS), collapse = " + "),
  "+ (1|thread_id) + (1|author)"
))
cat("\nFormula (full):\n"); print(f3_full); cat("\n")

m3_converged    <- TRUE
m3_author_dropped <- FALSE

m3 <- tryCatch({
  withCallingHandlers(
    glmer(f3_full, data = df5, family = binomial,
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
  if (!m3_converged)
    cat("Refitting without (1|author): convergence issue in full model\n")
  if (!is.na(author_var) && author_var < 0.01)
    cat(sprintf("Refitting without (1|author): variance = %.6f < 0.01\n", author_var))

  m3_author_dropped <- TRUE
  f3_reduced <- as.formula(paste(
    "y ~",
    paste(c("A_social", "A_hypothetical", CONTROLS), collapse = " + "),
    "+ (1|thread_id)"
  ))
  cat("\nFormula (reduced):\n"); print(f3_reduced); cat("\n")
  m3 <- glmer(f3_reduced, data = df5, family = binomial,
              control = glmerControl(optimizer = "bobyqa"))
}

saveRDS(list(model = m3, author_dropped = m3_author_dropped, author_var = author_var),
        file = "analysis/robustness_1to5_m3.rds")
cat("Saved: analysis/robustness_1to5_m3.rds\n")

# =============================================================================
# SUMMARY — A_social and A_hypothetical coefficients
# =============================================================================
cat("\n", strrep("=", 70), "\n", sep = "")
cat("COEFFICIENTS SUMMARY — A_social and A_hypothetical\n")
cat(strrep("=", 70), "\n", sep = "")

m1_mat <- matrix(m1_robust, ncol = 4)
rownames(m1_mat) <- rownames(m1_robust)
colnames(m1_mat) <- c("Estimate", "SE_robust", "z", "p")
m1_df <- as.data.frame(m1_mat)
m1_df$term <- rownames(m1_df)

m3_coefs <- as.data.frame(summary(m3)$coefficients)
colnames(m3_coefs) <- c("Estimate", "SE", "z", "p")
m3_coefs$term <- rownames(m3_coefs)

for (var in c("A_social", "A_hypothetical")) {
  r1 <- m1_df[m1_df$term == var, ]
  r3 <- m3_coefs[m3_coefs$term == var, ]
  cat(sprintf("\n%s\n", var))
  cat(sprintf("  M1 (robust): Est = %.4f  SE = %.4f  z = %.3f  p = %.4f\n",
              r1$Estimate, r1$SE_robust, r1$z, r1$p))
  cat(sprintf("  M3 (glmer):  Est = %.4f  SE = %.4f  z = %.3f  p = %.4f\n",
              r3$Estimate, r3$SE, r3$z, r3$p))
}

cat("\n=== robustness_1to5.R complete ===\n")
