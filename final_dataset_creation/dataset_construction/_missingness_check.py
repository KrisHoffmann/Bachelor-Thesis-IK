import polars as pl

df = pl.read_parquet(
    "/home/krishoffmann/projects/thesis_repository/final_dataset_creation"
    "/dataset_construction/outputs/stage7_comment_abstraction.parquet"
)
n = len(df)

def defined(col):
    return df[col].is_not_null() & ~df[col].is_nan()

soc = defined("A_social")
hyp = defined("A_hypothetical")

both     = (soc  &  hyp).sum()
soc_only = (soc  & ~hyp).sum()
hyp_only = (~soc &  hyp).sum()
neither  = (~soc & ~hyp).sum()

def pct(k):
    return k / n * 100

print(f"Total comments:                  {n:>10,}")
print()
print(f"Both A_social & A_hypothetical:  {both:>10,}  ({pct(both):.2f}%)")
print(f"A_social only:                   {soc_only:>10,}  ({pct(soc_only):.2f}%)")
print(f"A_hypothetical only:             {hyp_only:>10,}  ({pct(hyp_only):.2f}%)")
print(f"Neither defined:                 {neither:>10,}  ({pct(neither):.2f}%)")
print()
print(f"Sum check (== {n:,}): {both + soc_only + hyp_only + neither:,}")
