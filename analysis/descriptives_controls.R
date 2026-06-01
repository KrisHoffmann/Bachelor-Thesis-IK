# =============================================================================
# descriptives_controls.R
# Descriptive statistics for the nine surface controls used in M1.
#
# Prerequisites: run analysis/regression_prep.R first to generate
#                analysis/df3_prepped.rds
#
# Outputs (printed to console):
#   1. num_urls summary — count/pct of comments with >= 1 URL
#   2. Descriptives table (n, mean, SD, median, min, max) for raw controls
#   3. Same table for log1p-transformed controls (as entered in models)
# =============================================================================

# ── Load data ─────────────────────────────────────────────────────────────────
cat("\n--- Loading analysis/df3_prepped.rds ---\n")
df3 <- readRDS("analysis/df3_prepped.rds")
cat(sprintf("Loaded: %d rows, %d columns\n\n", nrow(df3), ncol(df3)))

# ── Raw control variables (before log1p) ─────────────────────────────────────
RAW_CONTROLS <- c(
  "noun_to_verb_ratio",
  "type_token_ratio",
  "flesch_kincaid",
  "hedge_density",
  "mean_sentence_length",
  "mean_word_length",
  "punctuation_density",
  "paragraph_count",
  "num_urls"
)

# ── Log1p controls (as used in models) ───────────────────────────────────────
LOG_CONTROLS <- paste0("log1p_", RAW_CONTROLS)

# =============================================================================
# 1. num_urls spotlight
# =============================================================================
cat(strrep("=", 70), "\n")
cat("num_urls SUMMARY\n")
cat(strrep("=", 70), "\n\n")

urls <- df3$num_urls
cat(sprintf("  Total observations : %d\n", length(urls)))
cat(sprintf("  Comments with >= 1 URL : %d (%.1f%%)\n",
            sum(urls >= 1, na.rm = TRUE),
            100 * mean(urls >= 1, na.rm = TRUE)))
cat(sprintf("  Comments with 0 URLs   : %d (%.1f%%)\n",
            sum(urls == 0, na.rm = TRUE),
            100 * mean(urls == 0, na.rm = TRUE)))
cat(sprintf("  Mean   : %.4f\n", mean(urls, na.rm = TRUE)))
cat(sprintf("  SD     : %.4f\n", sd(urls, na.rm = TRUE)))
cat(sprintf("  Median : %.4f\n", median(urls, na.rm = TRUE)))
cat(sprintf("  Max    : %d\n\n", max(urls, na.rm = TRUE)))

# =============================================================================
# 2. Descriptives — raw controls
# =============================================================================
cat(strrep("=", 70), "\n")
cat("DESCRIPTIVES — raw controls (pre-transform)\n")
cat(strrep("=", 70), "\n\n")

desc_raw <- do.call(rbind, lapply(RAW_CONTROLS, function(v) {
  if (!v %in% names(df3)) return(NULL)
  x <- df3[[v]]
  x <- x[is.finite(x)]
  data.frame(
    variable = v,
    n        = length(x),
    mean     = round(mean(x),   4),
    sd       = round(sd(x),     4),
    median   = round(median(x), 4),
    min      = round(min(x),    4),
    max      = round(max(x),    4),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}))

print(desc_raw, row.names = FALSE)

# =============================================================================
# 3. Descriptives — log1p controls (as entered in models)
# =============================================================================
cat("\n")
cat(strrep("=", 70), "\n")
cat("DESCRIPTIVES — log1p controls (as entered in models)\n")
cat(strrep("=", 70), "\n\n")

missing_log <- setdiff(LOG_CONTROLS, names(df3))
if (length(missing_log) > 0) {
  cat(sprintf("NOTE: the following log1p columns are missing — did regression_prep.R run?\n"))
  cat(paste0("  ", missing_log, collapse = "\n"), "\n\n")
  LOG_CONTROLS <- intersect(LOG_CONTROLS, names(df3))
}

desc_log <- do.call(rbind, lapply(LOG_CONTROLS, function(v) {
  x <- df3[[v]]
  x <- x[is.finite(x)]
  data.frame(
    variable = v,
    n        = length(x),
    mean     = round(mean(x),   4),
    sd       = round(sd(x),     4),
    median   = round(median(x), 4),
    min      = round(min(x),    4),
    max      = round(max(x),    4),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}))

print(desc_log, row.names = FALSE)

cat("\n=== descriptives_controls.R complete ===\n")
