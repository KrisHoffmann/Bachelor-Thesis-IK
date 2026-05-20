# analysis/ — Phase D Regression Scripts

## Run order

**Step 1 — `regression_prep.R`** — load, assert, impute, JB tests, log1p transforms, save RDS
**Step 2 — `collinearity_check.R`** — VIF diagnostics, iterative drop, print final CONTROLS vector
**Step 3 — `models.R`** — paste CONTROLS output from step 2, then run clogit and glmer

---

## Script summaries

### `regression_prep.R`

Loads the 1:3 matched case-control dataset, validates it, and produces a clean prepped dataframe ready for modelling.

1. Loads `stage9_matched_1to3.parquet` from `DATA_DIR` (see below).
2. Runs four hard assertions: row count (23,096), case count (5,849), no nulls in `A_social`/`A_hypothetical`, presence of both `stratum_id` and `thread_id`.
3. Imputes `noun_to_verb_ratio` NAs with the in-sample corpus mean (comments with zero verbs have no ratio; mean imputation is preferred over a sentinel value).
4. Runs a Jarque-Bera normality test on each of the 12 continuous linguistic controls.
5. Applies `log1p()` to any control where the JB test rejects normality (p < 0.05). Transformed columns are named `log1p_{varname}`; originals are kept.
6. Writes `analysis/jb_diagnostics.csv` (variable, JB stat, p-value, reject flag).
7. Writes `analysis/df3_prepped.rds` (full prepped dataframe, RDS format to preserve column types).

**After running this script:** open `jb_diagnostics.csv` to see which variables were transformed. The `CONTROLS` vector in `models.R` already has all 12 with `log1p_` prefixes — if any variable was *not* transformed, revert its prefix. Then run `collinearity_check.R` before touching `models.R`.

---

### `collinearity_check.R`

Runs VIF diagnostics on all 12 log1p-transformed controls to identify and remove collinear variables before modelling.

1. Loads `analysis/df3_prepped.rds`.
2. Fits a simple `glm(y ~ A_social + A_hypothetical + [all 12 controls], family=binomial)` — no strata, purely for VIF purposes.
3. Computes `car::vif()`, prints results sorted descending, saves to `analysis/vif_diagnostics.csv`.
4. Flags VIF > 5 (warn) and VIF > 10 (drop recommended).
5. Iteratively drops the single highest-VIF control above 10 and refits until all remaining controls are ≤ 10, printing each iteration.
6. Prints the final clean `CONTROLS` vector formatted to copy-paste directly into `models.R`.

Expected collinearity (anticipate these drops):
- `log1p_num_tokens` / `log1p_num_word_tokens` — near-identical length measures
- `log1p_num_sentences`, `log1p_mean_sentence_length`, `log1p_num_tokens` — correlated length proxies
- `log1p_flesch_kincaid` — a formula of `log1p_mean_word_length` and `log1p_mean_sentence_length`

**After running this script:** copy the printed `CONTROLS <- c(...)` block into `models.R`, replacing the existing vector, then run `models.R`.

---

### `models.R`

Fits the two active models (M1, M3) and prints the core thesis table.

- **Model 1** — Conditional logistic regression (`survival::clogit`), grouped by `strata(stratum_id)`, with cluster-robust SEs clustered at `thread_id`. This is the primary model.
- **Model 2** — Placeholder only. Implementation held pending supervisor sign-off on model type (hierarchical clogit vs. mediation). See Phase D handoff §3.
- **Model 3** — GLMM robustness check (`lme4::glmer`, binomial), random effects `(1|thread_id) + (1|author)`. If the author variance is < 0.01 or the model fails to converge, it automatically refits with `(1|thread_id)` only and documents the decision.

Outputs:
- `analysis/model1_results.rds` — M1 model object, robust coeftest, clustered vcov
- `analysis/model3_results.rds` — M3 model object, author-drop flag, author variance
- `analysis/thesis_table_main.csv` — core thesis table (log-odds and ORs for `A_social` and `A_hypothetical` across M1 and M3)

---

## Data path

At the top of `regression_prep.R`:

```r
DATA_DIR <- "C:/Users/kris/Documents/Thesis_R"
```

Change this if the parquet files move. `models.R` reads from `analysis/df3_prepped.rds` and does not need `DATA_DIR`.

---

## stratum_id vs thread_id — do not conflate these

These two columns exist in the dataset and serve entirely different roles:

| Column | What it identifies | Used for |
|---|---|---|
| `stratum_id` | One matched set: one case + its 3 (or 5) matched controls | `strata(stratum_id)` in `clogit` — this is the matching grouping variable |
| `thread_id` | A subreddit thread (submission) | Cluster-robust SEs in M1; random effect `(1|thread_id)` in M3 |

**Why this matters:** multiple strata can belong to the same thread (a thread can contain both the case comment and unrelated control comments drawn from the same thread). Using `strata(thread_id)` in `clogit` would silently mis-specify the model by grouping on the wrong unit — the matched-set structure would be destroyed and the conditional likelihood would be computed incorrectly. Always use `strata(stratum_id)` for the conditional logistic grouping.
