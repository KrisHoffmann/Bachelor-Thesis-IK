# =============================================================================
# exploratory_ts_descriptives.R
# Exploratory appendix — descriptive summary of all four abstraction indices
# on the FULL pre-matching inference corpus (stage8_comment_full.parquet,
# 598,272 rows).
#
# NO inferential tests. Descriptives only.
#
# Prerequisites: none (reads directly from the repo outputs directory).
# =============================================================================

# ── Packages ──────────────────────────────────────────────────────────────────
if (!requireNamespace("arrow", quietly = TRUE)) install.packages("arrow")
if (!requireNamespace("dplyr", quietly = TRUE)) install.packages("dplyr")

library(arrow)
library(dplyr)

# ── Path ──────────────────────────────────────────────────────────────────────
PARQUET <- "C:/Users/kris/Documents/Thesis_rcode/Bachelor-Thesis-IK/analysis/outputs/stage8_comment_full.parquet"

# Delta/award indicator: column "y" (1 = delta awarded, 0 = not awarded)
DELTA_COL  <- "y"

# Four abstraction indices
INDEX_COLS <- c("A_temporal", "A_spatial", "A_social", "A_hypothetical")

# =============================================================================
# 1. Load full corpus
# =============================================================================
cat("\n", strrep("=", 70), "\n", sep = "")
cat("LOADING FULL INFERENCE CORPUS\n")
cat(strrep("=", 70), "\n", sep = "")

cat(sprintf("\nReading: %s\n", PARQUET))
df <- arrow::read_parquet(PARQUET)

cat(sprintf("nrow: %d\n", nrow(df)))
cat(sprintf("ncol: %d\n", ncol(df)))
cat("\nColumn names:\n")
print(names(df))

if (nrow(df) != 598272) {
  cat(sprintf("NOTE: expected 598,272 rows, got %d\n", nrow(df)))
}

# =============================================================================
# 2. Overall descriptives for all four indices
# =============================================================================
cat("\n", strrep("=", 70), "\n", sep = "")
cat("OVERALL DESCRIPTIVES — all four abstraction indices\n")
cat(strrep("=", 70), "\n", sep = "")

n_total <- nrow(df)

desc_all <- lapply(INDEX_COLS, function(col) {
  x     <- df[[col]]
  x_def <- x[!is.na(x) & is.finite(x)]
  n_def <- length(x_def)
  data.frame(
    index       = col,
    n_total     = n_total,
    n_defined   = n_def,
    pct_defined = round(100 * n_def / n_total, 2),
    mean        = round(mean(x_def),   4),
    median      = round(median(x_def), 4),
    sd          = round(sd(x_def),     4),
    min         = round(min(x_def),    4),
    max         = round(max(x_def),    4),
    stringsAsFactors = FALSE
  )
})

desc_tbl <- do.call(rbind, desc_all)
rownames(desc_tbl) <- NULL
cat("\n")
print(desc_tbl, row.names = FALSE)

# =============================================================================
# 3. A_temporal and A_spatial split by delta/award indicator (y)
# =============================================================================
cat("\n", strrep("=", 70), "\n", sep = "")
cat("BY-GROUP DESCRIPTIVES — A_temporal and A_spatial (y=1 awarded vs y=0 not)\n")
cat(strrep("=", 70), "\n", sep = "")

cat(sprintf("\nDelta indicator column: '%s'\n", DELTA_COL))
cat("Value counts:\n")
print(table(df[[DELTA_COL]], useNA = "ifany"))

for (col in c("A_temporal", "A_spatial")) {
  cat(sprintf("\n--- %s by %s ---\n", col, DELTA_COL))

  grp_tbl <- df %>%
    dplyr::select(delta = dplyr::all_of(DELTA_COL),
                  idx   = dplyr::all_of(col)) %>%
    dplyr::filter(!is.na(idx) & is.finite(idx)) %>%
    dplyr::group_by(delta) %>%
    dplyr::summarise(
      n_defined = dplyr::n(),
      mean      = round(mean(idx),   4),
      median    = round(median(idx), 4),
      sd        = round(sd(idx),     4),
      .groups   = "drop"
    ) %>%
    dplyr::mutate(group = ifelse(delta == 1, "delta awarded (y=1)", "no delta (y=0)")) %>%
    dplyr::select(group, n_defined, mean, median, sd)

  print(as.data.frame(grp_tbl), row.names = FALSE)
}

# =============================================================================
cat("\n", strrep("=", 70), "\n", sep = "")
cat("NOTE: no inferential tests run. Descriptives only.\n")
cat(strrep("=", 70), "\n", sep = "")
cat("\n=== exploratory_ts_descriptives.R complete ===\n")
