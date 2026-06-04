# =============================================================================
# clt_fit_interaction.R
# CLT Fit-Account Interaction Test
#
# STATUS: STUB — OP-level abstraction index data not yet available.
#
# WHAT THIS SCRIPT WOULD TEST
# ---------------------------
# The "construal fit" account of persuasion predicts that a comment is most
# persuasive when its level of abstraction MATCHES the OP's construal level.
# This is tested by joining OP-post abstraction indices to the matched
# case-control sample and fitting three additional conditional logistic models
# that extend the primary model (M1 in models.R).
#
# WHAT DATA ARE NEEDED
# --------------------
# A file containing ONE ROW PER THREAD (keyed on thread_id) with at minimum:
#
#   thread_id   — matches the thread_id column in stage9_matched_1to3.parquet
#   A_Soc_OP    — social abstraction index of the OP selftext (A_social from
#                 the pipeline, but computed over the original post body, not
#                 comments). Value in [0, 1] or NaN if no salient sentences.
#   A_H_OP      — hypothetical abstraction index of the OP selftext (A_hypothetical
#                 from the pipeline, same convention).
#
# How to produce this file
# ------------------------
# 1. Identify OP selftext for each thread. In the raw CMV dump this is the
#    'selftext' field of submission objects (not comments). These can be
#    extracted from the Pushshift submissions file or from the PRAW-fetched
#    metadata already collected during data acquisition.
#
# 2. Run the existing BERT-based sentence classifier pipeline on the OP
#    selftexts. Use the same aggregate_indices.py logic:
#      A_d = mean(label_d for sentences where label_d in {0, 1})
#      A_d = NaN if no such sentence on dimension d
#
# 3. Save the resulting per-thread index file as, e.g.:
#      final_dataset_creation/dataset_construction/outputs/stage7_op_abstraction.parquet
#    Columns: thread_id, A_Soc_OP (= A_social), A_H_OP (= A_hypothetical).
#    Rename A_social -> A_Soc_OP and A_hypothetical -> A_H_OP to avoid column
#    name collisions when joining to the comment-level matched sample.
#
# MODELS TO FIT (once data are available)
# ----------------------------------------
# All three models extend the M1 formula from models.R:
#
#   base formula:
#     y ~ A_social + A_hypothetical +
#         log1p_noun_to_verb_ratio + log1p_type_token_ratio +
#         log1p_flesch_kincaid + log1p_hedge_density +
#         log1p_mean_sentence_length + log1p_mean_word_length +
#         log1p_punctuation_density + log1p_paragraph_count +
#         log1p_num_urls +
#         strata(stratum_id)
#
#   Model F1 (OP main effects):
#     base + A_Soc_OP + A_H_OP
#     Tests: does OP construal level independently predict delta, above and
#     beyond comment construal? If significant, the OP's framing affects
#     susceptibility to persuasion.
#
#   Model F2 (interaction / fit):
#     base + A_Soc_OP + A_H_OP + A_social:A_Soc_OP + A_hypothetical:A_H_OP
#     The fit account predicts POSITIVE interaction coefficients: persuasion
#     is maximised when comment and OP construal levels are aligned (both high
#     or both low abstraction on the same dimension).
#
#   Model F3 (continuous mismatch):
#     base + abs(A_social - A_Soc_OP) + abs(A_hypothetical - A_H_OP)
#     Negative coefficients on the distance terms would support the fit
#     account: smaller mismatch -> higher delta odds.
#
# All models use:
#   - survival::clogit(..., method = "efron")
#   - sandwich::vcovCL(..., cluster = ~thread_id) for clustered SEs
#   - lmtest::coeftest for printed coefficient table
#
# STRATA WARNING
# ---------------
# Strata that lack a matching OP record (A_Soc_OP = NaN or no join match)
# must be DROPPED entirely before fitting, because clogit requires a complete
# outcome contrast within each stratum. Print how many strata are dropped and
# verify the remaining sample still has enough power.
#
# OUTPUT
# -------
# analysis/output/fit_interaction_results.txt — plain-text tables for all
# three models (log-odds + OR format matching models.R output).
#
# =============================================================================

set.seed(42)

cat("=============================================================================\n")
cat("clt_fit_interaction.R — STUB\n")
cat("=============================================================================\n")
cat("\n")
cat("This script requires OP-level abstraction indices (A_Soc_OP, A_H_OP)\n")
cat("joined on thread_id to the matched case-control sample.\n")
cat("\n")
cat("No such file was found in the repository during the Step 1 exploration:\n")
cat("  Checked: final_dataset_creation/dataset_construction/outputs/\n")
cat("  Checked: outputs/\n")
cat("  Result:  No OP-level index file found; no is_op or row_type flag in\n")
cat("           the comment-level datasets.\n")
cat("\n")
cat("See the header comments above for:\n")
cat("  - What data are needed and how to produce them.\n")
cat("  - The exact model formulas (F1, F2, F3) to fit.\n")
cat("  - Output format and file location.\n")
cat("\n")
cat("Once the OP index file is available, replace this stub with the full\n")
cat("implementation described in the comments above and re-run.\n")
cat("\n")
cat("=== clt_fit_interaction.R (stub) complete ===\n")
