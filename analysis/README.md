# analysis/ — Phase D Regression Scripts

## Run order

**Step 1 — `regression_prep.R`** (always run first)
**Step 2 — `models.R`** (reads the output of step 1)

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

**After running this script:** open `jb_diagnostics.csv` and update the `CONTROLS` vector at the top of `models.R` — replace variable names with their `log1p_` versions wherever `reject_H0 == TRUE`.

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
