# =============================================================================
# export_controls.R
# Extract coefficient table for the nine surface controls from M1.
# Reads analysis/model1_results.rds (produced by models.R).
# Prints: term, estimate, robust SE, z, p.
# =============================================================================

res <- readRDS("analysis/model1_results.rds")

# robust_coeftest is a coeftest matrix: rows = terms, cols = Estimate/Std. Error/z/Pr(>|z|)
ct <- res$robust_coeftest

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

rows <- ct[CONTROLS, , drop = FALSE]

out <- data.frame(
  term      = rownames(rows),
  estimate  = rows[, "Estimate"],
  robust_se = rows[, "Std. Error"],
  z         = rows[, "z value"],
  p         = rows[, "Pr(>|z|)"],
  stringsAsFactors = FALSE
)

cat(sprintf("\n%-35s %9s %9s %7s %9s\n",
            "term", "estimate", "robust_se", "z", "p"))
cat(strrep("-", 75), "\n")
for (i in seq_len(nrow(out))) {
  cat(sprintf("%-35s %9.4f %9.4f %7.3f %9.4f\n",
              out$term[i], out$estimate[i], out$robust_se[i],
              out$z[i], out$p[i]))
}
cat("\n")
